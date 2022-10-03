from pathlib import Path

import pytest

from srctools import sndscript


def test_split() -> None:
    """Test float splitting logic."""
    enum_func = sndscript.Pitch.__getitem__

    assert sndscript.split_float('4.5', enum_func, 4.0, 'testing') == (4.5, 4.5)
    assert sndscript.split_float('notfloat', enum_func, 4.0, 'testing') == (4.0, 4.0)
    assert sndscript.split_float('1.2, 3.4', enum_func, 4.0, 'testing') == (1.2, 3.4)

    assert sndscript.split_float(
        ' Pitch_low ', enum_func, 4.0, '',
    ) == (sndscript.Pitch.PITCH_LOW, sndscript.Pitch.PITCH_LOW)

    assert sndscript.split_float(
        ' pITch_hiGh, pitCH_loW', enum_func, 4.0, '',
    ) == (sndscript.Pitch.PITCH_HIGH, sndscript.Pitch.PITCH_LOW)

    assert sndscript.split_float(' PItch_hiGH, 3.5', enum_func, 4.0, '') == (sndscript.Pitch.PITCH_HIGH, 3.5)
    assert sndscript.split_float('3.5, PItch_LOw ', enum_func, 4.0, '') == (3.5, sndscript.Pitch.PITCH_LOW)


def test_join() -> None:
    """Test float re-joining."""
    assert sndscript.join_float((4.5, 4.5)) == '4.5'
    assert sndscript.join_float((1.2, 3.4)) == '1.2, 3.4'

    assert sndscript.join_float((sndscript.Pitch.PITCH_LOW, sndscript.Pitch.PITCH_LOW)) == 'PITCH_LOW'
    assert sndscript.join_float((sndscript.VOL_NORM, 4.5)) == 'VOL_NORM, 4.5'
    assert sndscript.join_float((8.28, sndscript.VOL_NORM)) == '8.28, VOL_NORM'


@pytest.mark.parametrize('fname, loop', [
    ('sound_noloop.wav', False),
    ('sound_loop.wav', True),
])
def test_looping_check(fname: str, loop: bool, datadir: Path) -> None:
    """A very basic test of wav_is_looped, test two sample files."""
    with open(datadir / fname, 'rb') as f:
        assert sndscript.wav_is_looped(f) is loop
