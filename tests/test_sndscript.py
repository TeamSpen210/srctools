"""Test the soundscript module"""
import re

from pytest_datadir.plugin import LazyDataDir
import pytest

from srctools.keyvalues import Keyvalues
from srctools import sndscript


def test_split() -> None:
    """Test float splitting logic."""
    enum_func = sndscript.Pitch.__getitem__

    assert sndscript.split_float('4.5', enum_func, 4.0) == (4.5, 4.5)
    assert sndscript.split_float('notfloat', enum_func, 4.0) == (4.0, 4.0)
    assert sndscript.split_float('1.2, 3.4', enum_func, 4.0) == (1.2, 3.4)

    assert sndscript.split_float(
        ' Pitch_low ', enum_func, 4.0
    ) == (sndscript.Pitch.PITCH_LOW, sndscript.Pitch.PITCH_LOW)

    assert sndscript.split_float(
' pITch_hiGh, pitCH_loW', enum_func, 4.0,
    ) == (sndscript.Pitch.PITCH_HIGH, sndscript.Pitch.PITCH_LOW)

    assert sndscript.split_float(
        ' PItch_hiGH, 3.5', enum_func, 4.0,
    ) == (sndscript.Pitch.PITCH_HIGH, 3.5)
    assert sndscript.split_float(
        '3.5, PItch_LOw ', enum_func, 4.0
    ) == (3.5, sndscript.Pitch.PITCH_LOW)


def test_split_parse() -> None:
    """Test behaviour of parse_split_float."""
    kv = Keyvalues('World.TestSound', [
        Keyvalues('working', 'pitch_low, 4.5'),
        Keyvalues('a_block', [
            Keyvalues('a', '1'),
            Keyvalues('b', '2'),
        ])
    ])
    enum_func = sndscript.Pitch.__getitem__

    assert sndscript.parse_split_float(
        kv, 'working', enum_func, 4.0,
    ) == (sndscript.Pitch.PITCH_LOW, 4.5)

    assert sndscript.parse_split_float(
        kv, 'missing', enum_func, 4.0,
    ) == (4.0, 4.0)

    with pytest.raises(ValueError, match=re.escape(
        'Keyvalues block used for "a_block" option in "World.TestSound" sound!'
    )):
        sndscript.parse_split_float(kv, 'a_block', enum_func, 5.0, )


def test_join() -> None:
    """Test float re-joining."""
    assert sndscript.join_float((4.5, 4.5)) == '4.5'
    assert sndscript.join_float((1.2, 3.4)) == '1.2, 3.4'

    assert sndscript.join_float((sndscript.Pitch.PITCH_LOW, sndscript.Pitch.PITCH_LOW)) == 'PITCH_LOW'
    assert sndscript.join_float((sndscript.VOL_NORM, 4.5)) == 'VOL_NORM, 4.5'
    assert sndscript.join_float((8.28, sndscript.VOL_NORM)) == '8.28, VOL_NORM'


def test_soundchar_parse() -> None:
    """Test parsing sound characters out of filenames."""
    SoundChars = sndscript.SoundChars
    assert SoundChars.from_fname('sound/regular.wav') == (
        SoundChars.none, "sound/regular.wav")
    assert SoundChars.from_fname('#)sound/some_sound.wav') == (
        SoundChars.dry_mix | SoundChars.spatial_stereo, "sound/some_sound.wav")
    assert SoundChars.from_fname('*><^@sound\\some_SOund.wav') == (
        SoundChars.stream | SoundChars.directional | SoundChars.doppler | SoundChars.dist_variant
        | SoundChars.omni, "sound\\some_SOund.wav")

    assert SoundChars.from_fname('*test.wav') == (SoundChars.stream, 'test.wav')
    assert SoundChars.from_fname('?test.wav') == (SoundChars.user_vox, 'test.wav')
    assert SoundChars.from_fname('!test.wav') == (SoundChars.sentence, 'test.wav')
    assert SoundChars.from_fname('#test.wav') == (SoundChars.dry_mix, 'test.wav')
    assert SoundChars.from_fname('>test.wav') == (SoundChars.doppler, 'test.wav')
    assert SoundChars.from_fname('<test.wav') == (SoundChars.directional, 'test.wav')
    assert SoundChars.from_fname('^test.wav') == (SoundChars.dist_variant, 'test.wav')
    assert SoundChars.from_fname('@test.wav') == (SoundChars.omni, 'test.wav')
    assert SoundChars.from_fname(')test.wav') == (SoundChars.spatial_stereo, 'test.wav')
    assert SoundChars.from_fname('(test.wav') == (SoundChars.dir_stereo, 'test.wav')
    assert SoundChars.from_fname('}test.wav') == (SoundChars.fast_pitch, 'test.wav')
    assert SoundChars.from_fname('$test.wav') == (SoundChars.subtitled, 'test.wav')
    assert SoundChars.from_fname('&test.wav') == (SoundChars.hrtf_force, 'test.wav')
    assert SoundChars.from_fname('~test.wav') == (SoundChars.hrtf, 'test.wav')
    assert SoundChars.from_fname('`test.wav') == (SoundChars.hrtf_blend, 'test.wav')
    assert SoundChars.from_fname('+test.wav') == (SoundChars.radio, 'test.wav')
    assert SoundChars.from_fname('%test.wav') == (SoundChars.music, 'test.wav')


def test_soundchar_join() -> None:
    """Test reassembling sound characters."""
    SoundChars = sndscript.SoundChars
    assert str(SoundChars.dir_stereo | SoundChars.omni) == '@('
    assert str(SoundChars.omni | SoundChars.dir_stereo | SoundChars.hrtf) == '@(~'
    assert str(SoundChars.music | SoundChars.spatial_stereo) == ')%'
    assert str(SoundChars.stream | SoundChars.directional
               | SoundChars.doppler | SoundChars.dist_variant) == '*><^'

    assert str(SoundChars.stream) == '*'
    assert str(SoundChars.user_vox) == '?'
    assert str(SoundChars.sentence) == '!'
    assert str(SoundChars.dry_mix) == '#'
    assert str(SoundChars.doppler) == '>'
    assert str(SoundChars.directional) == '<'
    assert str(SoundChars.dist_variant) == '^'
    assert str(SoundChars.omni) == '@'
    assert str(SoundChars.spatial_stereo) == ')'
    assert str(SoundChars.dir_stereo) == '('
    assert str(SoundChars.fast_pitch) == '}'
    assert str(SoundChars.subtitled) == '$'
    assert str(SoundChars.hrtf_force) == '&'
    assert str(SoundChars.hrtf) == '~'
    assert str(SoundChars.hrtf_blend) == '`'
    assert str(SoundChars.radio) == '+'
    assert str(SoundChars.music) == '%'


def test_parse() -> None:
    """Test parsing a basic soundscript."""
    kv = Keyvalues.root(Keyvalues('some.Sound', [
        Keyvalues('channel', 'CHAN_VoICE'),
        Keyvalues('soundlevel', 'sndLVL_85db, 0.4'),
        Keyvalues('volume', '0.85, 0.95'),
        Keyvalues('wave', ')util/some_sound.wav'),
    ]))
    sound_dict = sndscript.Sound.parse(kv)
    assert len(sound_dict) == 1
    sound = sound_dict['some.sound']
    assert sound.sounds == [')util/some_sound.wav']
    assert sound.channel is sndscript.Channel.VOICE
    assert sound.level == (sndscript.Level.SNDLVL_85dB, 0.4)
    assert sound.volume == (0.85, 0.95)
    assert sound.force_v2 is False
    assert not sound.stack_start
    assert not sound.stack_stop
    assert not sound.stack_update


def test_parse_opstacks() -> None:
    """Test parsing a v2 soundscript, with operator stacks."""


@pytest.mark.parametrize('fname, loop', [
    ('sound_noloop.wav', False),
    ('sound_loop.wav', True),
])
def test_looping_check(
    fname: str, loop: bool,
    lazy_datadir: LazyDataDir,
) -> None:
    """A very basic test of wav_is_looped, test two sample files."""
    with open(lazy_datadir / fname, 'rb') as f:
        assert sndscript.wav_is_looped(f) is loop
