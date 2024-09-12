"""Test the VTF library."""
from typing import Generator
from array import array
from io import BytesIO
from pathlib import Path
import colorsys

from PIL import Image, ImageChops
from pytest_regressions.file_regression import FileRegressionFixture
import pytest

from srctools import vtf as vtf_mod
from srctools.vtf import VTF, ImageFormats


# A few formats aren't implemented by us/Valve.
FORMATS = [
    fmt for fmt in ImageFormats
    if fmt.name not in ["NONE", "P8", "RGBA16161616", "RGBA16161616F", "ATI1N", "ATI2N"]
]


def compare_img(obtained_fname: Path, expected_fname: Path) -> None:
    """Compare two images to ensure they are the same."""
    obtained = Image.open(obtained_fname)
    expected = Image.open(expected_fname)
    assert obtained.size == expected.size
    # If abs(a-b) == 0, images are the same.
    extrema = ImageChops.difference(obtained, expected).getextrema()
    if any(mins > 0 or maxes > 0 for mins, maxes in extrema):
        pytest.fail(f"{obtained_fname} and {expected_fname} do not match: {extrema!r}")


# noinspection PyProtectedMember
@pytest.fixture(params=["cython", "python"], ids=str.title)
def cy_py_format_funcs(request) -> Generator[str, None, None]:
    """Test against either the Cython or Python functions."""
    orig = vtf_mod._format_funcs
    kind: str = request.param
    try:
        module = getattr(vtf_mod, f"_{kind[:2]}_format_funcs")
        vtf_mod._format_funcs = module
        yield kind
    finally:
        vtf_mod._format_funcs = orig


@pytest.fixture(scope="session")
def sample_image() -> Image.Image:
    """Construct a sample image to test with."""
    img = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    for y in range(64):
        for x in range(4):
            lux = 255 - (y * 4 + x)
            img.putpixel((x, y), (lux, 0, 0, 255))
            img.putpixel((7-x, y), (0, lux, 0, 255))
            img.putpixel((56+x, y), (0, 0, lux, 255))
            img.putpixel((63-x, y), (255, 255, 255, lux))

    for x in range(8, 64-8):
        hue = (x - 8.0) / 48.0
        for y in range(64):
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, (1.0+y)/64.0)
            img.putpixel((x, y), (round(255*r), round(255*g), round(255*b), 255))
    return img


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name.lower())
def test_save(
    cy_py_format_funcs: str,
    fmt: ImageFormats,
    sample_image: Image.Image,
    file_regression: FileRegressionFixture,
) -> None:
    """Test saving as the specified format."""
    if cy_py_format_funcs == "python" and fmt.name.startswith(("DXT", "ATI")):
        pytest.xfail("DXT/ATI compression not implemented in Python code.")

    vtf = VTF(
        64, 64,
        fmt=fmt,
        # Default DXT1 is not implemented in Python mode.
        # Use this instead so it doesn't fail.
        thumb_fmt=ImageFormats.RGB888,
    )
    vtf.get().copy_from(sample_image.tobytes())

    buf = BytesIO()
    vtf.save(buf)
    file_regression.check(
        buf.getvalue(),
        binary=True,
        extension=".vtf",
        basename=f"test_save_{cy_py_format_funcs}_{fmt.name.lower()}"
    )


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name.lower())
def test_load(
    cy_py_format_funcs: str,
    fmt: ImageFormats,
    datadir: Path,
    file_regression: FileRegressionFixture,
) -> None:
    """Test loading the specified format.

    These samples were created using VTFEdit Reloaded.
    """
    with open(datadir / f"sample_{fmt.name.lower()}.vtf", "rb") as f:
        vtf = VTF.read(f)
        assert vtf.format is fmt
        vtf.load()
        img = vtf.get().to_PIL()

    buf = BytesIO()
    img.save(buf, "png")

    file_regression.check(
        buf.getvalue(),
        binary=True,
        extension=".png",
        basename=f"test_load_{cy_py_format_funcs}_{fmt.name.lower()}",
        check_fn=compare_img,
    )


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name.lower())
def test_load_header_only(
    fmt: ImageFormats,
    datadir: Path,
) -> None:
    """Test loading only the header."""
    with open(datadir / f"sample_{fmt.name.lower()}.vtf", "rb") as f:
        vtf = VTF.read(f, header_only=True)
        assert vtf.format is fmt
    # Check loading doesn't need the stream, it's not using them.
    img = vtf.get().to_PIL()
    # Check image is 64x64 and fully black.
    assert img.size == (64, 64)
    assert img.getextrema() == ((0, 0), (0, 0), (0, 0), (255, 255))


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name.lower())
def test_save_bad_size(
    cy_py_format_funcs: str,
    fmt: ImageFormats,
) -> None:
    """Test that the format functions produce errors if the buffers are incorrect."""
    if cy_py_format_funcs == "python" and fmt.name.startswith(("DXT", "ATI")):
        pytest.xfail("DXT/ATI compression not implemented in Python code.")

    with pytest.raises(BufferError):  # Wrong pixel size.
        vtf_mod._format_funcs.save(
            fmt,
            array('B'),
            bytearray(fmt.frame_size(32, 32)),
            32, 32,
        )
    with pytest.raises(BufferError):  # Wrong data size.
        vtf_mod._format_funcs.save(
            fmt,
            array('B', [0, 0, 0, 0xFF]) * (32 * 32),
            bytearray(),
            32, 32,
        )


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda fmt: fmt.name.lower())
def test_load_bad_size(
    cy_py_format_funcs: str,
    fmt: ImageFormats,
) -> None:
    """Test that the format functions produce errors if the buffers are incorrect."""
    with pytest.raises(BufferError):  # Wrong pixel size.
        vtf_mod._format_funcs.load(
            fmt,
            array('B'),
            bytes(fmt.frame_size(32, 32)),
            32, 32,
        )
    with pytest.raises(BufferError):  # Wrong data size.
        vtf_mod._format_funcs.load(
            fmt,
            array('B', [0, 0, 0, 0xFF]) * (32 * 32),
            b'',
            32, 32,
        )
