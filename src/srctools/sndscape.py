"""Parses soundscape files, which define sets of soundscripts/raw sounds to play."""
from enum import Enum

from typing import Union, Optional, NoReturn

import attrs

from . import conv_int, conv_float
from .math import Vec
from .keyvalues import Keyvalues
from .sndscript import (
    Volume, VOL_NORM, Pitch, Level, LevelInterval, Interval,
    join_float, parse_split_float,
)
from .types import FileWText


__all__ = ['PosType', 'Soundscape', 'SubScape', 'LoopSound', 'RandSound']


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
    volume: Interval[Volume]
    pitch: Interval[Pitch]
    level: LevelInterval
    no_restore: bool  #: If true, this rule is discarded if reloading from a save.

    @classmethod
    def _parse_pos(cls, kv: Keyvalues) -> Union[PosType, int, Vec]:
        """Parse the position option."""
        if 'origin' in kv:
            return Vec.from_str(kv['origin'])

        pos = kv['position', None]
        if pos is None:
            return PosType.AMBIENT
        elif pos.casefold() == 'random':
            return PosType.RANDOM
        else:
            return conv_int(pos)

    def export(self, file: FileWText) -> None:
        """Write these common parameters to a file."""
        if self.position is PosType.RANDOM:
            file.write('\t\tposition random\n')
        elif self.position is PosType.AMBIENT:
            pass  # Default
        elif isinstance(self.position, int):
            file.write(f'\t\tposition {self.position}\n')
        else:
            file.write(f'\t\tposition "{self.position}"\n')
        if self.no_restore:
            file.write('\t\tsuppress_on_restore 1\n')
        if self.volume != (1.0, 1.0) and self.volume != (VOL_NORM, VOL_NORM):
            file.write(f'\t\tvolume "{join_float(self.volume)}"\n')
        if self.pitch != (100, 100):
            file.write(f'\t\tpitch "{join_float(self.pitch)}"\n')
        if self.level != (Level.SNDLVL_NORM, Level.SNDLVL_NORM):
            file.write(f'\t\tsoundlevel "{Level.join_interval(self.level)}"\n')


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
        time: tuple[float, float] = parse_split_float(kv, 'time', no_enum, 0.0)

        return cls(
            pitch=Pitch.parse_interval_kv(kv),
            volume=Volume.parse_interval_kv(kv),
            level=Level.parse_interval_kv(kv),
            position=cls._parse_pos(kv),
            no_restore=kv.bool('suppress_on_restore'),
            time=time,
            sounds=sounds,
        )

    def export(self, file: FileWText) -> None:
        """Write this block to a file."""
        file.write('\tplayrandom\n\t\t{\n')
        file.write(f'\t\ttime "{join_float(self.time)}"\n')
        super().export(file)
        file.write('\t\trndwave\n\t\t\t{\n')
        for snd in self.sounds:
            file.write(f'\t\t\twave "{snd}"\n')
        file.write('\t\t\t}\n')
        file.write('\t\t}\n')


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
            volume=Volume.parse_interval_kv(kv),
            level=Level.parse_interval_kv(kv),
            position=cls._parse_pos(kv),
            radius=kv.float('radius', 0.0),
            no_restore=kv.bool('suppress_on_restore'),
        )

    def export(self, file: FileWText) -> None:
        """Write this block to a file."""
        file.write('\tplaylooping\n\t\t{\n')
        file.write(f'\t\twave "{self.sound}"\n')
        super().export(file)
        if self.radius != 0.0:
            file.write(f'\t\tradius "{self.radius}"\n')
        file.write('\t\t}\n')


@attrs.define(kw_only=True)
class SubScape:
    """A sub-soundscape entry, which allows overriding some values."""
    name: str  #: The name of the soundscape to play
    volume: Interval[Volume]
    pos_offset: int  #: This is added to any integral positions in the soundscape.
    pos_override: Optional[int]  #: If set, overrides all positions to this one.
    ambient_pos_override: Optional[int]  #: If set, overrides ambient sounds to use this position.

    @classmethod
    def parse(cls, kv: Keyvalues) -> 'SubScape':
        """Parse a playsoundscape entry."""
        # If pos override is set, it's the default for ambient pos override.
        pos_override = kv.int('positionoverride', None)
        ambient_pos_override = kv.int('ambientpositionoverride', pos_override)
        return cls(
            name=kv['name'],
            volume=Volume.parse_interval_kv(kv),
            pos_offset=kv.int('position', 0),
            pos_override=pos_override,
            ambient_pos_override=ambient_pos_override,
        )

    def export(self, file: FileWText) -> None:
        """Write this block to a file."""
        file.write('\tplaysoundscape\n\t\t{\n')
        file.write(f'\t\tname "{self.name}"\n')
        if self.volume != (1.0, 1.0) and self.volume != (VOL_NORM, VOL_NORM):
            file.write(f'\t\tvolume "{join_float(self.volume)}"\n')
        if self.pos_offset != 0:
            file.write(f'\t\tposition "{self.pos_offset}"\n')
        if self.pos_override is not None:
            file.write(f'\t\tpositionoverride "{self.pos_override}"\n')
        # This defaults to match the regular override, so no need to define if they match.
        if self.ambient_pos_override is not None and self.ambient_pos_override != self.pos_override:
            file.write(f'\t\tambientpositionoverride "{self.ambient_pos_override}"\n')
        file.write('\t\t}\n')


@attrs.define(kw_only=True)
class Soundscape:
    """A single soundscape definition."""
    name: str
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
        scape = cls(name=kv.real_name)
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

    def export(self, file: FileWText) -> None:
        """Write a soundscape to a file.

        Pass a file-like object open for text writing.
        """
        file.write(f'"{self.name}"\n\t{{\n')
        # Add a gap between soundscape options, and each type of sound,
        # but only if there's something before.
        gap = False
        if self.fadetime != 0.0:
            gap = True
            file.write(f'\tfadetime "{self.fadetime}"\n')
        if self.soundmixer is not None:
            gap = True
            file.write(f'\tsoundmixer "{self.soundmixer}"\n')
        if self.dsp is not None:
            gap = True
            file.write(f'\tdsp {self.dsp}\n')
        if self.dsp_spatial is not None:
            gap = True
            file.write(f'\tdsp_spatial {self.dsp_spatial}\n')
        if self.dsp_volume != 1.0:
            gap = True
            file.write(f'\tdsp_volume "{self.dsp_volume}"\n')

        if self.loop_sounds:
            if gap:
                file.write('\n')
            gap = True
            for loop in self.loop_sounds:
                loop.export(file)
        if self.rand_sounds:
            if gap:
                file.write('\n')
            gap = True
            for rand in self.rand_sounds:
                rand.export(file)
        if self.children:
            if gap:
                file.write('\n')
            gap = True
            for child in self.children:
                child.export(file)

        file.write('\t}\n')
