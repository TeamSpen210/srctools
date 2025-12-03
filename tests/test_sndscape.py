"""Test soundscape parsing."""
import io

from pytest_datadir.plugin import LazyDataDir
from dirty_equals import IsList, HasAttributes
from pytest_regressions.file_regression import FileRegressionFixture

from srctools import Keyvalues, Vec
from srctools.sndscript import Level, VOL_NORM, Pitch
from srctools.sndscape import Soundscape, PosType, LoopSound, RandSound, SubScape


def test_parse(lazy_datadir: LazyDataDir) -> None:
    """Test parsing a sample soundscape."""
    with open(lazy_datadir / 'sample_scape.txt') as f:
        kv = Keyvalues.parse(f)
    scapes = Soundscape.parse(kv)
    assert len(scapes) == 1
    scape = scapes['samples.simplescape']

    assert scape.name == 'samples.SimpleScape'
    assert scape.dsp == 83
    assert scape.dsp_spatial is None
    assert scape.dsp_volume == 1.0
    assert scape.fadetime == 0.75

    assert scape.rand_sounds == IsList(
        RandSound(
            position=5,
            volume=(0.8, 1.275),
            pitch=(50, 253),
            level=(Level.SNDLVL_TALKING, Level.SNDLVL_TALKING),
            no_restore=False,
            time=(0.5, 0.92),
            sounds=["ambient/wind1.wav", "ambient/wind2.wav", "ambient.Windy"],
        )
    )
    assert scape.loop_sounds == IsList(
        LoopSound(
            position=PosType.RANDOM,
            volume=(VOL_NORM, VOL_NORM),
            pitch=(Pitch.PITCH_LOW, Pitch.PITCH_HIGH),
            level=(66, 70),
            no_restore=True,
            radius=8192.85,
            sound='music/epic_music.mp3',
        ),
        LoopSound(
            position=Vec(0, -29.5, 1928),
            volume=(VOL_NORM, VOL_NORM),
            pitch=(Pitch.PITCH_NORM, Pitch.PITCH_NORM),
            level=(10, 10),
            no_restore=False,
            radius=0.0,
            sound='npc/whispers.wav'
        ),
    )
    assert scape.children == IsList(
        SubScape(
            name="samples.graveYard",
            volume=(VOL_NORM, VOL_NORM),
            pos_offset=0,
            pos_override=4,
            ambient_pos_override=5,
        ),
        SubScape(
            name="samples.beach_common",
            volume=(0.3, 0.4),
            pos_offset=1,
            pos_override=None,
            ambient_pos_override=None,
        ),
        SubScape(
            name="samples.birds",
            volume=(10.0, 11.0),
            pos_offset=0,
            pos_override=6,
            ambient_pos_override=6,
        ),
    )


def test_roundtrip(lazy_datadir: LazyDataDir, file_regression: FileRegressionFixture) -> None:
    """Test exporting, by resaving our test file."""
    with open(lazy_datadir / 'sample_scape.txt') as f:
        kv = Keyvalues.parse(f)
    scapes = Soundscape.parse(kv)

    buf = io.StringIO()
    for scape in scapes.values():
        scape.export(buf)
    file_regression.check(
        buf.getvalue(),
        extension='.txt',
    )
