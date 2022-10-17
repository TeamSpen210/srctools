from pathlib import Path

import pytest

from srctools import Keyvalues, sndscript


def test_split() -> None:
    """Test float splitting logic."""
    prop = Keyvalues('World.TestSound', [
        Keyvalues('num', '4.5'),
        Keyvalues('unparsable', 'notfloat'),
        Keyvalues('twonum', '1.2, 3.4'),

        Keyvalues('single_enum', ' Pitch_low '),
        Keyvalues('dual_enum', ' pITch_hiGh, pitCH_loW'),
        Keyvalues('enum_num', ' PItch_hiGH, 3.5'),
        Keyvalues('num_enum', '3.5, PItch_LOw '),
        Keyvalues('a_block', [
            Keyvalues('a', '1'),
            Keyvalues('b', '2'),
        ])
    ])
    enum_func = sndscript.Pitch.__getitem__

    assert sndscript.split_float(prop, 'num', enum_func, 4.0) == (4.5, 4.5)
    assert sndscript.split_float(prop, 'unparsable', enum_func, 4.0) == (4.0, 4.0)
    assert sndscript.split_float(prop, 'twonum', enum_func, 4.0) == (1.2, 3.4)

    assert sndscript.split_float(
        prop, 'single_enum', enum_func, 4.0
    ) == (sndscript.Pitch.PITCH_LOW, sndscript.Pitch.PITCH_LOW)

    assert sndscript.split_float(
        prop, 'dual_enum', enum_func, 4.0,
    ) == (sndscript.Pitch.PITCH_HIGH, sndscript.Pitch.PITCH_LOW)

    assert sndscript.split_float(prop, 'enum_num', enum_func, 4.0) == (sndscript.Pitch.PITCH_HIGH, 3.5)
    assert sndscript.split_float(prop, 'num_enum', enum_func, 4.0) == (3.5, sndscript.Pitch.PITCH_LOW)

    with pytest.raises(
        ValueError, match='Keyvalues block used for "a_block" option in "World.TestSound" sound!',
    ):
        sndscript.split_float(prop, 'a_block', enum_func, 5.0, )


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
