"""Functions for reading/writing VTF data."""
# We don't implement DXT saving, since that's highly complicated (and would be very slow).
# Wherever possible, use memoryview slicing to copy channels all in one go. This is much faster
# than a loop.

from typing import TYPE_CHECKING, NewType, Callable, Dict, Iterable, List, Optional, Tuple
from typing_extensions import TypeAlias, Buffer
import array


if TYPE_CHECKING:  # Avoid an import cycle.
    from srctools.vtf import FilterMode, ImageFormats
else:
    ImageFormats = 'ImageFormats'
    FilterMode = 'FilterMode'

ROView = NewType('ROView', memoryview)
RWView = NewType('RWView', ROView)
Array: TypeAlias = 'array.array[int]'
SaveFunc: TypeAlias = Callable[[Array, RWView, int, int], None]
LoadFunc: TypeAlias = Callable[[Array, ROView, int, int], None]
_SAVE: Dict[ImageFormats, SaveFunc] = {}
_LOAD: Dict[ImageFormats, LoadFunc] = {}


def ppm_convert(pixels: Array, width: int, height: int, bg: Optional[Tuple[int, int, int]]) -> bytes:
    """Convert a frame into a PPM-format bytestring, for passing to tkinter.

    If bg is set, this is the background we composite into. Otherwise, we just strip the alpha.
    """
    header = b'P6 %i %i 255\n' % (width, height)
    img_off = len(header)
    pix_count = width * height
    buffer = bytearray(img_off + 3 * pix_count)
    # Memoryviews to avoid making temp objects.
    view_src = memoryview(pixels)
    view_dest = memoryview(buffer)

    view_dest[0:img_off] = header

    if bg is not None:
        r, g, b = bg
        for offset in range(width * height):
            a = pixels[4 * offset + 3] / 255.0
            inv_a = 1.0 - a
            buffer[img_off + 3*offset] = int(pixels[4*offset + 0] * a + inv_a * r)
            buffer[img_off + 3*offset + 1] = int(pixels[4*offset + 1] * a + inv_a * g)
            buffer[img_off + 3*offset + 2] = int(pixels[4*offset + 2] * a + inv_a * b)
    else:
        # Copying in 3 slices means we can skip the loop over every pixel.
        view_dest[img_off:img_off + 4*pix_count:3] = view_src[::4]
        view_dest[img_off+1:img_off + 4*pix_count+1:3] = view_src[1::4]
        view_dest[img_off+2:img_off + 4*pix_count+2:3] = view_src[2::4]

    return bytes(buffer)


def alpha_flatten(pixels: Array, buffer: bytearray, width: int, height: int, bg: Optional[Tuple[int, int, int]]) -> bytes:
    """Flatten the image down to RGB, by removing the alpha channel.

    If bg is set, this is the background we composite into. Otherwise we
    just strip the alpha.
    """
    pix_count = width * height

    if bg is not None:
        r, g, b = bg
        for offset in range(width * height):
            a = pixels[4 * offset + 3] / 255.0
            inv_a = 1.0 - a
            buffer[3 * offset] = int(pixels[4 * offset + 0] * a + inv_a * r)
            buffer[3 * offset + 1] = int(pixels[4 * offset + 1] * a + inv_a * g)
            buffer[3 * offset + 2] = int(pixels[4 * offset + 2] * a + inv_a * b)
    else:
        view_src = memoryview(pixels)
        view_dest = memoryview(buffer)
        view_dest[0:4*pix_count:3] = view_src[::4]
        view_dest[1:4*pix_count+1:3] = view_src[1::4]
        view_dest[2:4*pix_count+2:3] = view_src[2::4]

    return bytes(buffer)


def upsample(bits: int, data: int) -> int:
    """Stretch bits worth of data to fill the byte.

    This is done by duplicating the MSB to fill the remaining space.
    """
    return data | (data >> bits)


def decomp565(a: int, b: int) -> Tuple[int, int, int]:
    """Decompress 565-packed data into RGB triplets."""
    return (
        upsample(5, (a & 0b00011111) << 3),
        upsample(6, ((b & 0b00000111) << 5) | ((a & 0b11100000) >> 3)),
        upsample(5, b & 0b11111000),
    )


def compress565(r: int, g: int, b: int) -> Tuple[int, int]:
    """Compress an RGB triplet into 565-packed data."""
    # RRRRRGGG GGGBBBBB
    return (
        (g << 3) & 0b11100000 | (b >> 3),
        (r & 0b11111000) | (g >> 5),
    )


def init(formats: Iterable[ImageFormats]) -> None:
    """Create a mapping from formats to functions."""
    glob = globals()
    for fmt in formats:
        try:
            _LOAD[fmt] = glob['load_' + fmt.name.casefold()]
        except KeyError:
            pass
        try:
            _SAVE[fmt] = glob['save_' + fmt.name.casefold()]
        except KeyError:
            pass


def load(
    fmt: ImageFormats,
    pixels: Array,
    data: Buffer,
    width: int, height: int,
) -> None:
    """Load pixels from data in the given format."""
    if memoryview(pixels).nbytes != 4 * width * height:
        raise BufferError(
            f"Incorrect pixel array size. Expected {4 * width * height} bytes, "
            f"got {memoryview(pixels).nbytes} bytes."
        )
    view_data = ROView(memoryview(data))
    expected_size = fmt.frame_size(width, height)
    if view_data.nbytes != expected_size:
        raise BufferError(
            f"Incorrect data block size. Expected {expected_size} bytes, "
            f"got {view_data.nbytes} bytes."
        )
    try:
        func = _LOAD[fmt]
    except KeyError:
        raise NotImplementedError(f"Loading {fmt.name} not implemented!") from None
    func(pixels, view_data, width, height)


def save(fmt: ImageFormats, pixels: Array, data: Buffer, width: int, height: int) -> None:
    """Save pixels from data in the given format."""
    if memoryview(pixels).nbytes != 4 * width * height:
        raise BufferError(
            f"Incorrect pixel array size. Expected {4 * width * height} bytes, "
            f"got {memoryview(pixels).nbytes} bytes."
        )
    view_data = RWView(ROView(memoryview(data)))
    expected_size = fmt.frame_size(width, height)
    if view_data.nbytes != expected_size:
        raise BufferError(
            f"Incorrect data block size. Expected {expected_size} bytes, "
            f"got {view_data.nbytes} bytes."
        )
    try:
        func = _SAVE[fmt]
    except KeyError:
        raise NotImplementedError(f"Saving {fmt.name} not implemented!") from None
    func(pixels, view_data, width, height)


def scale_down(
    filt: FilterMode,
    src_width: int, src_height: int,
    width: int, height: int,
    src: Array, dest: Array,
) -> None:
    """Scale down the image to this smaller size.

    This is simplified for mipmap generation only:
    either dimension may be the same, or be scaled exactly half.
    """
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

    if filt.value in (0, 1, 2, 3):  # Nearest-neighbour.
        # 0 = upper-left, 3 = lower-right
        pos_off = [
            0, horiz_off,
            vert_off, vert_off + horiz_off,
        ][filt.value]
        for y in range(height):
            for x in range(width):
                off = 4 * (width * y + x)
                off2 = 4 * (per_row * y + per_column * x)
                dest[off:off+4] = src[off2 + pos_off: off2 + pos_off + 4]
    elif filt.value == 4:  # Bilinear
        for y in range(height):
            for x in range(width):
                off = 4 * (width * y + x)
                off2 = 4 * (per_row * y + per_column * x)
                for channel in (0, 1, 2, 3):
                    dest[off + channel] = (
                        src[off2 + channel] +
                        src[off2 + channel + horiz_off] +
                        src[off2 + channel + vert_off] +
                        src[off2 + channel + vert_off + horiz_off]
                    ) // 4
    else:
        raise ValueError(f"Unknown filter {filt}!")


def saveload_rgba(mode: str) -> Tuple[LoadFunc, SaveFunc]:
    """Make the RGB save and load functions."""
    r_off = mode.index('r')
    g_off = mode.index('g')
    b_off = mode.index('b')
    try:
        a_off = mode.index('a')
    except ValueError:
        def loader_rgb(pixels: Array, data: ROView, width: int, height: int) -> None:
            view_pix = memoryview(pixels)
            view_pix[0::4] = data[r_off::3]
            view_pix[1::4] = data[g_off::3]
            view_pix[2::4] = data[b_off::3]
            view_pix[3::4] = b'\xFF' * (width * height)

        def saver_rgb(pixels: Array, data: RWView, width: int, height: int) -> None:
            view_pix = memoryview(pixels)
            data[r_off::3] = view_pix[0::4]
            data[g_off::3] = view_pix[1::4]
            data[b_off::3] = view_pix[2::4]

        loader_rgb.__name__ = 'load_' + mode
        saver_rgb.__name__ = 'save_' + mode
        return loader_rgb, saver_rgb
    else:
        def loader_rgba(pixels: Array, data: ROView, width: int, height: int) -> None:
            view_pix = memoryview(pixels)
            view_pix[0::4] = data[r_off::4]
            view_pix[1::4] = data[g_off::4]
            view_pix[2::4] = data[b_off::4]
            view_pix[3::4] = data[a_off::4]

        def saver_rgba(pixels: Array, data: RWView, width: int, height: int) -> None:
            view_pix = memoryview(pixels)
            data[r_off::4] = view_pix[0::4]
            data[g_off::4] = view_pix[1::4]
            data[b_off::4] = view_pix[2::4]
            data[a_off::4] = view_pix[3::4]

        loader_rgba.__name__ = 'load_' + mode
        saver_rgba.__name__ = 'save_' + mode
        return loader_rgba, saver_rgba


load_rgba8888, save_rgba8888 = saveload_rgba('rgba')
load_bgra8888, save_bgra8888 = saveload_rgba('bgra')

# This is totally the wrong order, but it's how it's actually ordered.
load_argb8888, save_argb8888 = saveload_rgba('gbar')
load_abgr8888, save_abgr8888 = saveload_rgba('abgr')

load_rgb888, save_rgb888 = saveload_rgba('rgb')
load_bgr888, save_bgr888 = saveload_rgba('bgr')


# These semantically operate differently, but just have 4 channels.
load_uvlx8888, save_uvlx8888 = saveload_rgba('rgba')
load_uvwq8888, save_uvwq8888 = saveload_rgba('rgba')


def load_bgrx8888(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Strange - skip byte."""
    view_pix = memoryview(pixels)
    view_pix[0::4] = data[2::4]
    view_pix[1::4] = data[1::4]
    view_pix[2::4] = data[0::4]
    view_pix[3::4] = b'\xFF' * (width * height)


def save_bgrx8888(pixels: Array, data: RWView, width: int, height: int) -> None:
    """Strange - skip byte."""
    view_pix = memoryview(pixels)
    data[3::4] = b'\0' * (width * height)
    data[2::4] = view_pix[0::4]
    data[1::4] = view_pix[1::4]
    data[0::4] = view_pix[2::4]


def load_rgb565(pixels: Array, data: ROView, width: int, height: int) -> None:
    """RGB format, packed into 2 bytes by dropping LSBs."""
    for offset in range(width * height):
        r, g, b = decomp565(data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset] = r
        pixels[4 * offset + 1] = g
        pixels[4 * offset + 2] = b
        pixels[4 * offset + 3] = 255


def save_rgb565(pixels: Array, data: RWView, width: int, height: int) -> None:
    """RGB format, packed into 2 bytes by dropping LSBs."""
    for offset in range(width * height):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset],
            pixels[4 * offset + 1],
            pixels[4 * offset + 2],
        )


def load_bgr565(pixels: Array, data: ROView, width: int, height: int) -> None:
    """BGR format, packed into 2 bytes by dropping LSBs."""
    for offset in range(width * height):
        b, g, r = decomp565(data[2 * offset], data[2 * offset + 1])

        pixels[4 * offset] = r
        pixels[4 * offset + 1] = g
        pixels[4 * offset + 2] = b
        pixels[4 * offset + 3] = 255


def save_bgr565(pixels: Array, data: RWView, width: int, height: int) -> None:
    """BGR format, packed into 2 bytes by dropping LSBs."""
    for offset in range(width * height):
        data[2*offset], data[2 * offset + 1] = compress565(
            pixels[4 * offset + 2],
            pixels[4 * offset + 1],
            pixels[4 * offset],
        )


def load_bgra4444(pixels: Array, data: ROView, width: int, height: int) -> None:
    """BGRA format, only upper 4 bits. Bottom half is a copy of the top."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset+1] = (a & 0b11110000) | (a & 0b11110000) >> 4
        pixels[4 * offset+2] = (a & 0b00001111) | (a & 0b00001111) << 4
        pixels[4 * offset] = (b & 0b00001111) | (b & 0b00001111) << 4
        pixels[4 * offset+3] = (b & 0b11110000) | (b & 0b11110000) >> 4


def save_bgra4444(pixels: Array, data: RWView, width: int, height: int) -> None:
    """BGRA format, only upper 4 bits. Bottom half is a copy of the top."""
    for offset in range(width * height):
        r = pixels[4 * offset]
        g = pixels[4 * offset + 1]
        b = pixels[4 * offset + 2]
        a = pixels[4 * offset + 3]

        data[2 * offset] = (g & 0b11110000) | (b >> 4)
        data[2 * offset + 1] = (a & 0b11110000) | (r >> 4)


def load_bgra5551(pixels: Array, data: ROView, width: int, height: int) -> None:
    """BGRA format, 5 bits per color plus 1 bit of alpha."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset] = upsample(5, (b & 0b01111100) << 1)
        pixels[4 * offset+1] = upsample(5, (a & 0b11100000) >> 2 | (b & 0b00000011) << 6)
        pixels[4 * offset+2] = upsample(5, (a & 0b00011111) << 3)
        pixels[4 * offset+3] = 255 if b & 0b10000000 else 0


def save_bgra5551(pixels: Array, data: RWView, width: int, height: int) -> None:
    """BGRA format, 5 bits per color plus 1 bit of alpha."""
    for offset in range(width * height):
        r = pixels[4 * offset]
        g = pixels[4 * offset + 1]
        b = pixels[4 * offset + 2]
        a = pixels[4 * offset + 3]
        # GGGBBBBB  ARRRRRGG
        data[2 * offset + 0] = ((g << 2) & 0b11100000) | (b >> 3)
        data[2 * offset + 1] = (a & 0b10000000) | ((r >> 1) & 0b01111100) | (g >> 6)


def load_bgrx5551(pixels: Array, data: ROView, width: int, height: int) -> None:
    """BGR format, 5 bits per color, alpha ignored."""
    for offset in range(width * height):
        a = data[2 * offset]
        b = data[2 * offset + 1]
        pixels[4 * offset] = upsample(5, (b & 0b01111100) << 1)
        pixels[4 * offset+1] = upsample(5, (a & 0b11100000) >> 2 | (b & 0b00000011) << 6)
        pixels[4 * offset+2] = upsample(5, (a & 0b00011111) << 3)
        pixels[4 * offset+3] = 255


def save_bgrx5551(pixels: Array, data: RWView, width: int, height: int) -> None:
    """BGR format, 5 bits per color, alpha ignored."""
    for offset in range(width * height):
        r = pixels[4 * offset]
        g = pixels[4 * offset + 1]
        b = pixels[4 * offset + 2]
        # GGGBBBBB  XRRRRRGG
        data[2 * offset + 0] = ((g << 2) & 0b11100000) | (b >> 3)
        data[2 * offset + 1] = ((r >> 1) & 0b01111100) | (g >> 6)


def load_i8(pixels: Array, data: ROView, width: int, height: int) -> None:
    """I8 format, R=G=B"""
    view_pix = memoryview(pixels)
    view_dat = memoryview(data)
    view_pix[0::4] = view_dat
    view_pix[1::4] = view_dat
    view_pix[2::4] = view_dat
    view_pix[3::4] = b'\xff' * (width * height)


def save_i8(pixels: Array, data: RWView, width: int, height: int) -> None:
    """Save in greyscale."""
    for offset in range(width * height):
        data[offset] = (
            pixels[4 * offset] +
            pixels[4 * offset + 1] +
            pixels[4 * offset + 2]
        ) // 3


def load_ia88(pixels: Array, data: ROView, width: int, height: int) -> None:
    """I8 format, R=G=B + A"""
    view_pix = memoryview(pixels)
    view_dat = memoryview(data)
    view_pix[0::4] = view_pix[1::4] = view_pix[2::4] = view_dat[0::2]
    view_pix[3::4] = view_dat[1::2]


def save_ia88(pixels: Array, data: RWView, width: int, height: int) -> None:
    """I8 format, R=G=B + A"""
    for offset in range(width * height):
        data[2 * offset] = (
            pixels[4 * offset] +
            pixels[4 * offset + 1] +
            pixels[4 * offset + 2]
        ) // 3
    memoryview(data)[1::2] = memoryview(pixels)[3::4]

# ImageFormats.P8 is not implemented by Valve either.


def load_a8(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Single alpha bytes."""
    view_pix = memoryview(pixels)
    view_pix[:] = bytes(4 * width * height)
    view_pix[3::4] = data


def save_a8(pixels: Array, data: RWView, width: int, height: int) -> None:
    """Single alpha bytes."""
    data[:] = memoryview(pixels)[3::4]


def load_uv88(pixels: Array, data: ROView, width: int, height: int) -> None:
    """UV-only, which is mapped to RG."""
    view_pix = memoryview(pixels)
    view_pix[:] = b'\0\0\0\xFF' * (width * height)
    view_pix[0::4] = data[0::2]
    view_pix[1::4] = data[1::2]


def save_uv88(pixels: Array, data: RWView, width: int, height: int) -> None:
    """UV-only, which is mapped to RG."""
    view_pix = memoryview(pixels)
    data[0::2] = view_pix[0::4]
    data[1::2] = view_pix[1::4]


def load_rgb888_bluescreen(pixels: Array, data: ROView, width: int, height: int) -> None:
    """RGB format, with 'bluescreen' mode for alpha.

    Pure blue pixels are transparent.
    """
    for offset in range(width * height):
        r = data[3 * offset]
        g = data[3 * offset + 1]
        b = data[3 * offset + 2]
        if r == g == 0 and b == 255:
            pixels[4*offset] = pixels[4*offset+1] = 0
            pixels[4*offset+2] = pixels[4*offset+3] = 0
        else:
            pixels[4 * offset] = r
            pixels[4 * offset + 1] = g
            pixels[4 * offset + 2] = b
            pixels[4 * offset + 3] = 255


def save_rgb888_bluescreen(pixels: Array, data: RWView, width: int, height: int) -> None:
    """RGB format, with 'bluescreen' mode for alpha.

    Transparent pixels are made blue.
    """
    for offset in range(width * height):
        if pixels[4 * offset + 3] < 128:
            data[3 * offset] = 0
            data[3 * offset + 1] = 0
            data[3 * offset + 2] = 255
        else:
            data[3 * offset] = pixels[4 * offset]
            data[3 * offset + 1] = pixels[4 * offset + 1]
            data[3 * offset + 2] = pixels[4 * offset + 2]


def load_bgr888_bluescreen(pixels: Array, data: ROView, width: int, height: int) -> None:
    """BGR format, with 'bluescreen' mode for alpha.

    Pure blue pixels are transparent.
    """
    for offset in range(width * height):
        r = data[3 * offset + 2]
        g = data[3 * offset + 1]
        b = data[3 * offset]
        if r == g == 0 and b == 255:
            pixels[4*offset] = pixels[4*offset+1] = 0
            pixels[4*offset+2] = pixels[4*offset+3] = 0
        else:
            pixels[4 * offset] = r
            pixels[4 * offset + 1] = g
            pixels[4 * offset + 2] = b
            pixels[4 * offset + 3] = 255


def save_bgr888_bluescreen(pixels: Array, data: RWView, width: int, height: int) -> None:
    """BGR format, with 'bluescreen' mode for alpha.

    Transparent pixels are made blue.
    """
    for offset in range(width * height):
        if pixels[4 * offset + 3] < 128:
            data[3 * offset + 2] = 0
            data[3 * offset + 1] = 0
            data[3 * offset] = 255
        else:
            data[3 * offset + 2] = pixels[4 * offset]
            data[3 * offset + 1] = pixels[4 * offset + 1]
            data[3 * offset] = pixels[4 * offset + 2]


def load_dxt1(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Load compressed DXT1 data."""
    load_dxt1_impl(pixels, data, width, height, (0, 0, 0, 0xFF))


def load_dxt1_onebitalpha(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Load compressed DXT1 data, with an additional 1 bit of alpha squeezed in."""
    load_dxt1_impl(pixels, data, width, height, (0, 0, 0, 0))


def load_dxt1_impl(
    pixels: Array,
    data: ROView,
    width: int,
    height: int,
    black_color: Tuple[int, int, int, int],
) -> None:
    """Does the actual decompression."""
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in range(0, 4 * width * height, 4):
            pixels[offset] = 0
            pixels[offset + 1] = 0
            pixels[offset + 2] = 0
            pixels[offset + 2] = 0xFF
        return

    block_wid, mod = divmod(width, 4)
    if mod:
        block_wid += 1

    for block_y in range(0, height, 4):
        block_y //= 4
        for block_x in range(0, width, 4):
            block_x //= 4
            block_off = 8 * (block_wid * block_y + block_x)

            # First, load the 2 colors.
            c0r, c0g, c0b = decomp565(data[block_off], data[block_off+1])
            c1r, c1g, c1b = decomp565(data[block_off+2], data[block_off+3])

            # We store the lookup colors as bytes so we can directly copy them.

            # Equivalent to 16-bit comparison...
            if (
                data[block_off] > data[block_off+2] or
                data[block_off+1] > data[block_off+3]
            ):
                c2 = (
                    (2*c0b + c1b) // 3,
                    (2*c0g + c1g) // 3,
                    (2*c0r + c1r) // 3,
                    255
                )
                c3 = (
                    (c0b + 2*c1b) // 3,
                    (c0g + 2*c1g) // 3,
                    (c0r + 2*c1r) // 3,
                    255
                )
            else:
                c2 = (
                    (c0r + c1r) // 2,
                    (c0g + c1g) // 2,
                    (c0b + c1b) // 2,
                    255
                )
                c3 = black_color

            table = [
                (c0b, c0g, c0r, 255),
                (c1b, c1g, c1r, 255),
                c2,
                c3,
            ]
            dxt_color_table(
                pixels, data, table,
                block_off, block_wid,
                block_x, block_y,
            )


def dxt_color_table(
    pixels: Array,
    data: ROView,
    table: List[Tuple[int, int, int, int]],
    block_off: int,
    block_wid: int,
    block_x: int,
    block_y: int,
) -> None:
    """Decodes the actual colour table into pixels."""
    for y in range(4):
        byte = data[block_off + 4 + y]
        row = 16 * block_wid * (4 * block_y + y) + 16 * block_x
        (
            pixels[row + 12],
            pixels[row + 13],
            pixels[row + 14],
            pixels[row + 15],
        ) = table[(byte & 0b11000000) >> 6]
        (
            pixels[row + 8],
            pixels[row + 9],
            pixels[row + 10],
            pixels[row + 11],
        ) = table[(byte & 0b00110000) >> 4]
        (
            pixels[row + 4],
            pixels[row + 5],
            pixels[row + 6],
            pixels[row + 7],
        ) = table[(byte & 0b00001100) >> 2]
        (
            pixels[row + 0],
            pixels[row + 1],
            pixels[row + 2],
            pixels[row + 3],
        ) = table[byte & 0b00000011]


def dxt_alpha_table(
    pixels: Array,
    data: ROView,
    block_off: int,
    block_wid: int,
    block_x: int,
    block_y: int,
    layer: int,
) -> None:
    """Decode the DXT5 alpha block into pixels.

    This is split out for ATI1/2N support as well.
    """
    alpha0 = data[block_off]
    alpha1 = data[block_off + 1]
    if alpha0 >= alpha1:
        alpha_table = [
            alpha0,
            alpha1,
            (6 * alpha0 + 1 * alpha1) // 7,
            (5 * alpha0 + 2 * alpha1) // 7,
            (4 * alpha0 + 3 * alpha1) // 7,
            (3 * alpha0 + 4 * alpha1) // 7,
            (2 * alpha0 + 5 * alpha1) // 7,
            (1 * alpha0 + 6 * alpha1) // 7,
        ]
    else:
        alpha_table = [
            alpha0,
            alpha1,
            (4 * alpha0 + 1 * alpha1) // 5,
            (3 * alpha0 + 2 * alpha1) // 5,
            (2 * alpha0 + 3 * alpha1) // 5,
            (1 * alpha0 + 4 * alpha1) // 5,
            0,
            255
        ]
    # The alpha data is a 48-bit integer, where each 3 bits maps to an alpha
    # value.
    lookup = int.from_bytes(data[block_off + 2:block_off + (2 + 6)], 'little')
    for i in range(16):
        y, x = divmod(i, 4)
        pos = 16 * block_wid * (4 * block_y + y) + 4 * (4 * block_x + x)
        pixels[pos + layer] = alpha_table[(lookup >> (3 * i)) & 0b111]


def load_dxt3(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Load compressed DXT3 data."""
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in range(0, 4 * width * height, 4):
            pixels[offset] = 0
            pixels[offset + 1] = 0
            pixels[offset + 2] = 0
            pixels[offset + 2] = 0xFF
        return

    block_wid, mod = divmod(width, 4)
    if mod:
        block_wid += 1

    for block_y in range(0, height, 4):
        block_y //= 4
        for block_x in range(0, width, 4):
            block_x //= 4
            block_off = 16 * (block_wid * block_y + block_x)

            # First, load the 2 colors.
            c0r, c0g, c0b = decomp565(data[block_off + 8], data[block_off + 9])
            c1r, c1g, c1b = decomp565(data[block_off + 10], data[block_off + 11])

            table: List[Tuple[int, int, int, int]] = [
                (c0b, c0g, c0r, 255),
                (c1b, c1g, c1r, 255),
                (
                    (2 * c0b + c1b) // 3,
                    (2 * c0g + c1g) // 3,
                    (2 * c0r + c1r) // 3,
                    255,
                ),
                (
                    (c0b + 2 * c1b) // 3,
                    (c0g + 2 * c1g) // 3,
                    (c0r + 2 * c1r) // 3,
                    255,
                ),
            ]
            dxt_color_table(
                pixels, data, table,
                block_off+8, block_wid,
                block_x, block_y,
            )
            # Now add on the real alpha values.
            for off in range(8):
                byte = data[block_off + off]
                y, x = divmod(off * 2, 4)
                pos = 16 * block_wid * (4 * block_y + y) + 4 * (4 * block_x + x)
                # Combine the values twice, so we evenly cover the whole range.
                pixels[pos + 3] = byte & 0b00001111 | (byte & 0b00001111) << 4
                pixels[pos + 7] = byte & 0b11110000 | (byte & 0b11110000) >> 4


def load_dxt5(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Load compressed DXT5 data."""
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in range(0, 4 * width * height, 4):
            pixels[offset] = 0
            pixels[offset + 1] = 0
            pixels[offset + 2] = 0
            pixels[offset + 2] = 0xFF
        return

    block_wid, mod = divmod(width, 4)
    if mod:
        block_wid += 1

    for block_y in range(0, height, 4):
        block_y //= 4
        for block_x in range(0, width, 4):
            block_x //= 4
            block_off = 16 * (block_wid * block_y + block_x)

            # Now, load the colour blocks.
            c0r, c0g, c0b = decomp565(data[block_off + 8], data[block_off + 9])
            c1r, c1g, c1b = decomp565(data[block_off + 10], data[block_off + 11])

            table: List[Tuple[int, int, int, int]] = [
                (c0b, c0g, c0r, 127),
                (c1b, c1g, c1r, 127),
                (
                    (2 * c0b + c1b) // 3,
                    (2 * c0g + c1g) // 3,
                    (2 * c0r + c1r) // 3,
                    127
                ),
                (
                    (c0b + 2 * c1b) // 3,
                    (c0g + 2 * c1g) // 3,
                    (c0r + 2 * c1r) // 3,
                    127
                ),
            ]
            dxt_color_table(
                pixels, data, table,
                block_off+8, block_wid,
                block_x, block_y,
            )
            dxt_alpha_table(
                pixels, data,
                block_off, block_wid,
                block_x, block_y,
                3,  # Put into alpha pixels
            )

# Don't do the high-def 16-bit resolution.

# def load_rgba16161616f(pixels, offset, data, data_off):
#     """16-bit RGBA format - max resolution."""
#     pixels[offset] = data[data_off] << 8 + data[data_off+1]
#     pixels[offset + 1] = data[data_off+2] << 8 + data[data_off+3]
#     pixels[offset + 2] = data[data_off+4] << 8 + data[data_off+5]
#     pixels[offset + 3] = data[data_off+6] << 8 + data[data_off+7]


def load_ati2n(pixels: Array, data: ROView, width: int, height: int) -> None:
    """Load 'ATI2N' format data, also known as BC5.

    This uses two copies of the DXT5 alpha block for data.
    """
    if width < 4 or height < 4:
        # DXT format must be 4x4 at minimum. So just write black.
        # They still exist in small mipmaps.
        for offset in range(0, 4 * width * height, 4):
            pixels[offset] = 0
            pixels[offset + 1] = 0
            pixels[offset + 2] = 0
            pixels[offset + 2] = 0xFF
        return

    block_wid, mod = divmod(width, 4)
    if mod:
        block_wid += 1

    for block_y in range(0, height, 4):
        block_y //= 4
        for block_x in range(0, width, 4):
            block_x //= 4
            block_off = 16 * (block_wid * block_y + block_x)

            dxt_alpha_table(
                pixels, data,
                block_off, block_wid,
                block_x, block_y,
                0,  # R channel
            )
            dxt_alpha_table(
                pixels, data,
                block_off + 8, block_wid,
                block_x, block_y,
                1,  # G channel
            )
    # Blank out the unused channels.
    for offset in range(width * height):
        pixels[4 * offset + 2] = 0
        pixels[4 * offset + 3] = 255
