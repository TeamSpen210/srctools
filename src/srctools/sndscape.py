"""Parses soundscape files, which define sets of soundscripts/raw sounds to play."""
from enum import Enum

from typing import Union, Optional, NoReturn

import attrs

from srctools import Vec, conv_int, conv_float
from srctools.keyvalues import Keyvalues
from srctools.sndscript import VOLUME, Pitch, Level, parse_split_float


def no_enum(_: str, /) -> NoReturn:
    """A function which never returns an enum, inteded for parse_split_float."""
    raise ValueError


class PosType(Enum):
    """Type of sound position."""
    # Play from a random position near the player
    RANDOM = 'random'
    # Default behaviour, play from all locations.
    AMBIENT = 'ambient'


@attrs.define(kw_only=True)
class SoundRule:
    """Common attributes for both types of sounds."""
    # Specific position type, integral position defined by the ent, or a fixed location.
    position: Union[PosType, int, Vec]
    volume: tuple[Union[float, VOLUME], Union[float, VOLUME]]
    pitch: tuple[Union[float, Pitch], Union[float, Pitch]]
    level: tuple[Union[float, Level], Union[float, Level]]
    no_restore: bool  #: If true, this rule is discarded if reloading from a save.

    @classmethod
    def _parse_pos(cls, kv: Keyvalues) -> Union[PosType, int, Vec]:
        """Parse the position option."""
        if 'origin' in kv:
            return Vec.from_str(kv.value)

        pos = kv['position', None]
        if pos is None:
            return PosType.AMBIENT
        elif pos.casefold() == 'random':
            return PosType.RANDOM
        else:
            return conv_int(pos)


@attrs.define(kw_only=True)
class RandSound(SoundRule):
    """A playrandom entry in a soundscape."""

    time: tuple[float, float]
    sounds: list[str]

    @classmethod
    def parse(cls, kv: Keyvalues) -> 'RandSound':
        """Parse a playrandom entry."""
        sounds = []
        for child in kv.find_all('rndwave'):
            sounds.extend(child.as_array())
        time = parse_split_float(kv, 'time', no_enum, 0.0)

        return cls(
            pitch=Pitch.parse_interval_kv(kv),
            volume=VOLUME.parse_interval_kv(kv),
            level=Level.parse_interval_kv(kv),
            position=cls._parse_pos(kv),
            no_restore=kv.bool('suppress_on_restore'),
            time=time,
            sounds=sounds,
        )


@attrs.define(kw_only=True)
class LoopSound(SoundRule):
    """A playlooping entry in a soundscape."""
    radius: float
    sound: str

    @classmethod
    def parse(cls, kv: Keyvalues) -> 'LoopSound':
        """Parse a playlooping entry."""
        return cls(
            sound=kv['wave'],
            pitch=Pitch.parse_interval_kv(kv),
            volume=VOLUME.parse_interval_kv(kv),
            level=Level.parse_interval_kv(kv),
            position=cls._parse_pos(kv),
            radius=kv.float('radius', 0.0),
            no_restore=kv.bool('suppress_on_restore'),
        )


@attrs.define(kw_only=True)
class SubScape:
    """A sub-soundscape entry, which allows overriding some values."""
    name: str  #: The name of the soundscape to play
    volume: tuple[Union[float, VOLUME], Union[float, VOLUME]]
    pos_offset: int  #: This is added to any integral positions in the soundscape.
    pos_override: Optional[int]  #: If set, overrides all positions to this one.
    ambient_pos_override: Optional[int]  #: If set, overrides ambient sounds to use this position.

    @classmethod
    def parse(cls, kv: Keyvalues) -> 'SubScape':
        """Parse a playsoundscape entry."""


@attrs.define(kw_only=True)
class Soundscape:
    """A single soundscape definition."""
    rand_sounds: list[RandSound] = attrs.Factory(list)
    loop_sounds: list[LoopSound] = attrs.Factory(list)
    children: list[SubScape] = attrs.Factory(list)

    # Scalar options set by the scape.
    dsp: Optional[int] = None
    dsp_spatial: Optional[int] = None
    dsp_volume: float = 1.0
    fadetime: float = 0.0
    soundmixer: Optional[str] = None

    @classmethod
    def parse(cls, file: Keyvalues) -> dict[str, 'Soundscape']:
        """Parse all soundscape definitions in a file.

        This returns a dict mapping casefolded names to Sounds.
        """
        return {kv.name: cls.parse_one(kv) for kv in file}

    @classmethod
    def parse_one(cls, kv: Keyvalues) -> 'Soundscape':
        """Parse a single soundscape definition."""
        scape = cls()
        for child_kv in kv:
            if child_kv.name == 'playlooping':
                scape.loop_sounds.append(LoopSound.parse(child_kv))
            elif child_kv.name == 'playrandom':
                scape.rand_sounds.append(RandSound.parse(child_kv))
            elif child_kv.name == 'playsoundscape':
                scape.children.append(SubScape.parse(child_kv))
            elif child_kv.name == 'dsp':
                scape.dsp = conv_int(child_kv.value)
            elif child_kv.name == 'dsp_spatial':
                scape.dsp_spatial = conv_int(child_kv.value)
            elif child_kv.name == 'dsp_volume':
                scape.dsp_volume = conv_float(child_kv.value, 1.0)
            elif child_kv.name == 'fadetime':
                scape.fadetime = conv_float(child_kv.value, 0.0)
            elif child_kv.name == 'soundmixer':
                scape.soundmixer = child_kv.value
            else:
                raise ValueError(f'Unknown soundscape option:', child_kv.real_name)
        return scape
