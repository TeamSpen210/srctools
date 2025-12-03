"""Reads and writes Soundscripts."""
from typing import Callable, Optional, TypeVar, Union, TypeAlias

from collections.abc import Mapping
import enum
import struct
import re

import attrs

from . import conv_float, conv_int
from .keyvalues import Keyvalues, NoKeyError
from .types import FileRSeek, FileWText


__all__ = [
    'SoundChars', 'Pitch', 'Channel', 'Level', 'Volume', 'VOL_NORM',
    'Sound', 'wav_is_looped', 'parse_split_float', 'split_float', 'join_float',
    'SND_CHARS', 'Interval', 'LevelInterval',
]

EnumT = TypeVar('EnumT')
Interval: TypeAlias = tuple[Union[float, EnumT], Union[float, EnumT]]
LevelInterval: TypeAlias = tuple[Union[int, 'Level'], Union[int, 'Level']]


#: Possible sound characters.
#: deprecated: Use SoundChars instead.
SND_CHARS = '*@#<>^)}$!?'


class SoundChars(enum.Flag):
    """Flags to represent sound characters which can be added to the start of sound filenames.

    Despite comments in the SDK, more than 2 are permitted simultaneously.
    To re-assemble into a set of characters, call :py:obj:`str`.
    """
    none = 0  #: A filename with no characters included.

    #: Stream the sound from disc, discarded afterwards. Use for one-off dialogue files or
    # music, to not keep them in memory.
    stream = enum.auto()
    user_vox = enum.auto()  #: Marks player voice chat data, shouldn't ever be used.
    sentence = enum.auto()  #: Dialog from the NPC sentence system.
    dry_mix = enum.auto()  #: DSP FX is bypassed for this sound.
    doppler = enum.auto()  #: Doppler encoded stereo wav: left wav (incoming) and right wav (outgoing).
    #: Stereo wav has direction cone: mix left wav (front facing)
    #: with right wav (rear facing) based on sound facing direction
    directional = enum.auto()
    dist_variant = enum.auto()  #: Distance variant encoded stereo wav (left is close, right is far)
    #: Non-directional - sound appears to play from everywhere,
    #: but still has distance volume falloff.
    omni = enum.auto()
    spatial_stereo = enum.auto()  #: Spatialised stereo wav
    dir_stereo = enum.auto()  #: Directional stereo wav (like doppler)
    fast_pitch = enum.auto()  #: Forces low quality, non-interpolated pitch shift
    subtitled = enum.auto()  #: Indicates subtitles were forced on.

    hrtf_force = enum.auto()  #: CSGO+ only. Enables HRTF for all players including the owner.
    hrtf = enum.auto()  #: CSGO+ only. Enables HRTF for non-owners.
    #: CSGO+ only. Enables HRTF for non-owner players, fading to stereo instead if close.
    hrtf_blend = enum.auto()
    radio = enum.auto()  #: CSGO+ only. Used for 'radio' sounds tha are played without spatialisation.
    music = enum.auto()  #: CSGO+ only. Used for main menu music.

    @classmethod
    def from_fname(cls, filename: str) -> tuple['SoundChars', str]:
        """Parse sound characters out of a filename, then return both."""
        flag = cls.none
        i = 0
        for i, char in enumerate(filename):
            try:
                flag |= CHAR_TO_FLAG[char]
            except KeyError:
                break
        return flag, filename[i:]

    def __str__(self) -> str:
        return ''.join([
            char
            for char, flag in CHAR_TO_FLAG.items()
            if flag in self
        ])


CHAR_TO_FLAG = {
    '*': SoundChars.stream,
    '?': SoundChars.user_vox,
    '!': SoundChars.sentence,
    '#': SoundChars.dry_mix,
    '>': SoundChars.doppler,
    '<': SoundChars.directional,
    '^': SoundChars.dist_variant,
    '@': SoundChars.omni,
    ')': SoundChars.spatial_stereo,
    '(': SoundChars.dir_stereo,
    '}': SoundChars.fast_pitch,
    '$': SoundChars.subtitled,
    '&': SoundChars.hrtf_force,
    '~': SoundChars.hrtf,
    '`': SoundChars.hrtf_blend,
    '+': SoundChars.radio,
    '%': SoundChars.music,
}
FLAG_TO_CHAR = {flag: char for char, flag in CHAR_TO_FLAG.items()}


class Pitch(float, enum.Enum):
    """The constants permitted for sound pitches."""
    PITCH_NORM = 100.0
    PITCH_LOW = 95.0
    PITCH_HIGH = 120.0

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse_interval_kv(self, kv: Keyvalues, key: str = 'pitch') -> Interval['Level']:
        """Parse an interval of pitches from a subkey of a keyvalues block."""
        return parse_split_float(kv, key, Pitch.__getitem__, 100)


class Volume(enum.Enum):
    """Special value, substitutes default volume (usually 1)."""
    VOL_NORM = 'VOL_NORM'

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        """We only have one value, and it's available globally - use that name."""
        return 'srctools.sndscript.VOL_NORM'

    @classmethod
    def parse_interval_kv(self, kv: Keyvalues, key: str = 'volume') -> Interval['Channel']:
        """Parse an interval of volumes from a subkey of a keyvalues block."""
        return parse_split_float(kv, key, Volume.__getitem__, 1.0)


VOL_NORM = Volume.VOL_NORM
VOLUME = Volume  # Deprecated alias

# Old compatibility values, replaced by soundlevel.
ATTENUATION: Mapping[str, float] = {
    'ATTN_NONE': 0.0,
    'ATTN_NORM': 0.8,
    'ATTN_IDLE': 2.0,
    'ATTN_STATIC': 1.25,
    'ATTN_RICOCHET': 1.5,
    'ATTN_GUNFIRE': 0.27,
}


class Channel(enum.Enum):
    """Different categories of sounds."""
    DEFAULT = "CHAN_AUTO"
    GUNFIRE = "CHAN_WEAPON"
    VOICE = "CHAN_VOICE"
    TF2_ANNOUNCER = "CHAN_VOICE2"
    ITEMS = "CHAN_ITEM"
    BODY = "CHAN_BODY"
    STREAMING = "CHAN_STREAM"
    CON_CMD = "CHAN_REPLACE"
    BACKGROUND = "CHAN_STATIC"
    PLAYER_VOICE = "CHAN_VOICE_BASE"

    #  CHAN_USER_BASE+<number>
    #  Custom channels can be defined here.

    def __str__(self) -> str:
        return self.value


class Level(enum.Enum):
    """Soundlevel constants - attenuation."""
    SNDLVL_NONE = 0
    SNDLVL_IDLE = 60
    SNDLVL_STATIC = 66
    SNDLVL_NORM = 75
    SNDLVL_TALKING = 80
    SNDLVL_GUNFIRE = 140

    def __str__(self) -> str:
        return self.name

    def __int__(self) -> int:
        return self.value

    @classmethod
    def parse(cls, text: str) -> Union[int, 'Level']:
        """Parse a soundlevel string, which might be a ``SNDLVL`` name or a number."""
        text = text.upper()
        try:
            return cls[text]
        except KeyError:
            pass
        # Game code doesn't actually validate the 'db' bit, just stops as soon as a non-num is present.
        if (match := re.match('SNDLVL_([0-9]+)', text)) is not None:
            num = int(match.group(1))
        else:
            num = conv_int(text, -1)
        if 0 <= num <= 180:
            return num
        else:
            return Level.SNDLVL_NORM

    @classmethod
    def parse_interval_kv(cls, kv: Keyvalues) -> LevelInterval:
        """Parse an interval of attenuations from the ``soundlevel`` and ``attenuation`` keys of a keyvalues block.

        Unlike others, this is rounded to an integer. Valve's code only supports the ``SNDLVL`` names
        for a single value, but we allow it for an interval.
        """
        if 'attenuation' in kv:
            atten_min: float
            atten_max: float
            atten_min, atten_max = parse_split_float(
                kv, 'attenuation',
                ATTENUATION.__getitem__,
                ATTENUATION['ATTN_IDLE'],
            )
            return (atten_to_level(atten_min), atten_to_level(atten_max))
        elif 'soundlevel' in kv:
            level_min, level_max = parse_split_float(
                kv, 'soundlevel',
                cls.parse,
                Level.SNDLVL_NORM,
            )
            # The game does parse as float then round down, so that's accurate.
            # We additionally allow SNDLVL constants in both positions.
            return (
                level_min if isinstance(level_min, Level) else int(level_min),
                level_max if isinstance(level_max, Level) else int(level_max),
            )
        else:
            return (Level.SNDLVL_NORM, Level.SNDLVL_NORM)

    @classmethod
    def join_interval(cls, interval: LevelInterval) -> str:
        """Convert an interval back to a string. Valve requires full integers for intervals."""
        low, high = interval
        if int(low) == int(high):
            if isinstance(low, Level):
                return str(low)  # SNDLVL_IDLE etc
            elif isinstance(high, Level):
                return str(high)  # (0, SNDLVL_NONE) -> name
            elif low % 5 == 0:  # Standard levels.
                return f'SNDLVL_{low}dB'
            else:
                return str(low)  # Can also be SNDLVL, but nicer to do this.
        else:
            if isinstance(low, Level):
                low = low.value
            if isinstance(high, Level):
                high = high.value
            return f'{low}, {high}'


SOUND_LEVELS: Mapping[str, Union[float, Level]] = {
    level.name.upper(): level
    for level in Level
} | {
    f'SNDLVL_{x}DB': x
    for x in range(181)
}


def parse_split_float(
    kv: Keyvalues,
    key: str,
    enum: Callable[[str], Union[float, EnumT]],
    default: Union[float, EnumT],
) -> Interval[EnumT]:
    """Parse pairs of float/enum values from keyvalues.

    A single number can be provided, producing the same value for low and high.

    :param kv: The keyvalues block for the sound, which ``key`` is then accessed from.
    :param key: The name of the keyvalue to look up inside ``kv``.
    :param enum: This is either an Enum with values to match text constants, or a converter function
        returning enums or raising ValueError, KeyError or IndexError.
    :param default: If either value or the whole string is unparsable, this default is used.
    """
    try:
        leaf_kv = kv.find_key(key)
    except NoKeyError:
        return (default, default)
    if leaf_kv.has_children():
        raise ValueError(f'Keyvalues block used for "{key}" option in "{kv.real_name}" sound!')
    return split_float(leaf_kv.value, enum, default)


def split_float(
    value: str,
    enum: Callable[[str], Union[float, EnumT]],
    default: Union[float, EnumT],
) -> Interval[EnumT]:
    """Handle values which can be a low, high pair of numbers or enum constants.

    A single number can be provided, producing the same value for low and high.

    :param value: The value to read.
    :param enum: This is either an Enum with values to match text constants, or a converter function
        returning enums or raising ValueError, KeyError or IndexError.
    :param default: If either value or the whole string is unparsable, this default is used.
    """
    if ',' in value:
        s_low, s_high = value.split(',')
        try:
            low = enum(s_low.strip().upper())
        except (LookupError, ValueError):
            low = conv_float(s_low, default)
        try:
            high = enum(s_high.strip().upper())
        except (LookupError, ValueError):
            high = conv_float(s_high, default)
        return low, high
    else:
        try:
            out = enum(value.strip().upper())
        except (LookupError, ValueError):
            out = conv_float(value, default)
        return out, out


def join_float(val: Interval[enum.Enum]) -> str:
    """Reverse split_float(). The two parameters should be stringifiable into floats/constants."""
    low, high = val
    if low == high:
        return str(low)
    else:
        return f'{low!s}, {high!s}'


def atten_to_level(attenuation: float) -> int:
    """Convert an old attenuation value to a soundlevel.

    See ATTN_TO_SNDLVL in :sdk-2025:`public/soundflags.h#L105`
    # TODO: Link that ^^
    """
    if attenuation:
        return int(50.0 + 20.0 / attenuation)
    else:
        return 0


class _WAVChunk:
    """To allow reading CUE points, we need a copy of the former chunk module.

    This is copied from the Python Standard Library, 3.11.
    We force little-endian, and to align to 2-byte boundaries.
    """
    def __init__(self, file: FileRSeek[bytes]) -> None:
        self.closed = False
        self.file = file
        self.chunkname = file.read(4)
        if len(self.chunkname) < 4:
            raise EOFError
        try:
            [self.chunksize] = struct.unpack_from('<L', file.read(4))
        except struct.error:
            raise EOFError from None
        self.size_read = 0

    def read(self, size: int = -1) -> bytes:
        """Read at most size bytes from the chunk.
        If size is omitted or negative, read until the end
        of the chunk.
        """
        if self.size_read >= self.chunksize:
            return b''
        if size < 0:
            size = self.chunksize - self.size_read
        if size > self.chunksize - self.size_read:
            size = self.chunksize - self.size_read
        data = self.file.read(size)
        self.size_read = self.size_read + len(data)
        if self.size_read == self.chunksize and (self.chunksize & 1):
            dummy = self.file.read(1)
            self.size_read = self.size_read + len(dummy)
        return data

    def skip(self) -> None:
        """Skip the rest of the chunk.
        If you are not interested in the contents of the chunk,
        this method should be called so that the file points to
        the start of the next chunk.
        """
        n = self.chunksize - self.size_read
        # maybe fix alignment
        if self.chunksize & 1:
            n = n + 1
        try:
            self.file.seek(n, 1)
        except (AttributeError, OSError):  # Cannot seek, manually read.
            while self.size_read < self.chunksize:
                n = min(8192, self.chunksize - self.size_read)
                skipped = self.read(n)
                if not skipped:
                    raise EOFError from None
        else:
            self.size_read = self.size_read + n


def wav_is_looped(file: FileRSeek[bytes]) -> bool:
    """Check if the provided wave file contains loop cue points.

    This code is partially copied from wave.Wave_read.initfp().
    """
    first = _WAVChunk(file)
    if first.chunkname != b'RIFF':
        raise ValueError('File does not start with RIFF id.')
    if first.read(4) != b'WAVE':
        raise ValueError('Not a WAVE file.')

    while True:
        try:
            chunk = _WAVChunk(file)
        except EOFError:
            return False
        if chunk.chunkname == b'cue ':
            return True
        chunk.skip()


@attrs.define(eq=False, init=False, repr=False)
class Sound:
    """Represents a single soundscript."""
    name: str
    sounds: list[str] = attrs.Factory(list)
    volume: Interval[Volume] = (VOL_NORM, VOL_NORM)
    channel: Union[int, Channel] = Channel.DEFAULT
    level: LevelInterval = (Level.SNDLVL_NORM, Level.SNDLVL_NORM)
    pitch: Interval[Pitch] = (Pitch.PITCH_NORM, Pitch.PITCH_NORM)

    _stack_start: Optional[Keyvalues] = None
    _stack_update: Optional[Keyvalues] = None
    _stack_stop: Optional[Keyvalues] = None
    force_v2: bool = False

    def __init__(
        self,
        name: str,
        sounds: list[str],
        volume: Union[Interval[Volume], float, Volume] = (VOL_NORM, VOL_NORM),
        channel: Union[int, Channel] = Channel.DEFAULT,
        level: Union[LevelInterval, int, Level] = (Level.SNDLVL_NORM, Level.SNDLVL_NORM),
        pitch: Union[Interval[Pitch], float, Pitch] = (Pitch.PITCH_NORM, Pitch.PITCH_NORM),

        # Operator stacks
        stack_start: Optional[Keyvalues] = None,
        stack_update: Optional[Keyvalues] = None,
        stack_stop: Optional[Keyvalues] = None,
        force_v2: bool = False,
    ) -> None:
        """Create a soundscript."""
        self.name = name
        self.sounds = sounds
        self.channel = channel
        self.force_v2 = force_v2

        if isinstance(volume, tuple):
            self.volume = volume
        else:
            self.volume = volume, volume

        if isinstance(level, tuple):
            self.level = level
        else:
            self.level = level, level

        if isinstance(pitch, tuple):
            self.pitch = pitch
        else:
            self.pitch = pitch, pitch

        self._stack_start = stack_start
        self._stack_update = stack_update
        self._stack_stop = stack_stop

    @property
    def stack_start(self) -> Keyvalues:
        """Initialise the stack if not already produced."""
        if self._stack_start is None:
            self._stack_start = Keyvalues('', [])
        return self._stack_start

    @stack_start.setter
    def stack_start(self, tree: Keyvalues) -> None:
        """Change the start stack to another tree."""
        self._stack_start = tree

    @property
    def stack_update(self) -> Keyvalues:
        """Initialise the stack if not already produced."""
        if self._stack_update is None:
            self._stack_update = Keyvalues('', [])
        return self._stack_update

    @stack_update.setter
    def stack_update(self, tree: Keyvalues) -> None:
        """Change the update stack to another tree."""
        self._stack_update = tree

    @property
    def stack_stop(self) -> Keyvalues:
        """Initialise the stack if not already produced."""
        if self._stack_stop is None:
            self._stack_stop = Keyvalues('', [])
        return self._stack_stop

    @stack_stop.setter
    def stack_stop(self, tree: Keyvalues) -> None:
        """Change the stop stack to another tree."""
        self._stack_stop = tree

    def __repr__(self) -> str:
        res = (
            f'{self.__class__.__name__}({self.name!r}, {self.sounds!r}, volume={self.volume!r}, '
            f'channel={self.channel!r}, level={self.level!r}, pitch={self.pitch!r}'
        )
        if self.force_v2 or self._stack_start or self._stack_update or self._stack_stop:
            res += (
                f', stack_start={self.stack_start!r}'
                f', stack_update={self.stack_update!r}'
                f', stack_stop={self.stack_stop!r})'
            )
        else:
            res += ')'
        return res

    @classmethod
    def parse(cls, file: Keyvalues) -> dict[str, 'Sound']:
        """Parses a soundscript file.

        This returns a dict mapping casefolded names to Sounds.
        """
        return {
            sound_kv.name: cls.parse_one(sound_kv)
            for sound_kv in file
        }

    @classmethod
    def parse_one(cls, sound_kv: Keyvalues) -> 'Sound':
        """Parse a single soundscript definition."""
        volume = Volume.parse_interval_kv(sound_kv)
        pitch = Pitch.parse_interval_kv(sound_kv)
        level = Level.parse_interval_kv(sound_kv)

        # Either 1 "wave", or multiple in "rndwave".
        wavs: list[str] = []
        for prop in sound_kv:
            if prop.name == 'wave':
                wavs.append(prop.value)
            elif prop.name == 'rndwave':
                for subprop in prop:
                    wavs.append(subprop.value)

        channel_str = sound_kv['channel', 'CHAN_AUTO'].upper()
        channel: Union[int, Channel]
        if channel_str.startswith('CHAN_'):
            channel = Channel(channel_str)
        else:
            channel = int(channel_str)

        sound_version = sound_kv.int('soundentry_version', 1)

        start_stack: Optional[Keyvalues]
        update_stack: Optional[Keyvalues]
        stop_stack: Optional[Keyvalues]
        if 'operator_stacks' in sound_kv:
            if sound_version == 1:
                raise ValueError(
                    'Operator stacks used with version '
                    f'less than 2 in "{sound_kv.real_name}"!'
                )

            start_stack, update_stack, stop_stack = (
                Keyvalues(stack_name, [
                    prop.copy()
                    for prop in
                    sound_kv.find_children('operator_stacks', stack_name)
                ])
                for stack_name in
                ['start_stack', 'update_stack', 'stop_stack']
            )
        else:
            start_stack = update_stack = stop_stack = None

        return Sound(
            sound_kv.real_name,
            wavs,
            volume,
            channel,
            level,
            pitch,
            start_stack,
            update_stack,
            stop_stack,
            sound_version == 2,
        )

    def export(self, file: FileWText) -> None:
        """Write a sound to a file.

        Pass a file-like object open for text writing.
        """
        file.write(f'"{self.name}"\n\t{{\n')
        file.write(f'\tchannel {self.channel}\n')
        file.write(f'\tsoundlevel {Level.join_interval(self.level)}\n')

        if self.volume != (1, 1):
            file.write(f'\tvolume {join_float(self.volume)}\n')
        if self.pitch != (100, 100):
            file.write(f'\tpitch {join_float(self.pitch)}\n')

        if len(self.sounds) != 1:
            file.write('\trndwave\n\t\t{\n')
            for wav in self.sounds:
                file.write(f'\t\twave "{wav}"\n')
            file.write('\t\t}\n')
        else:
            file.write(f'\twave "{self.sounds[0]}"\n')

        if self.force_v2 or self.stack_start or self.stack_stop or self.stack_update:
            file.write(
                '\t' 'soundentry_version 2\n'
                '\t' 'operator_stacks\n'
                '\t\t' '{\n'
            )
            if self.stack_start:
                file.write(
                    '\t\t' 'start_stack\n'
                    '\t\t\t' '{\n'
                )
                for kv in self.stack_start:
                    kv.serialise(file, start_indent='\t\t\t')
                file.write('\t\t\t}\n')
            if self.stack_update:
                file.write(
                    '\t\t' 'update_stack\n'
                    '\t\t\t' '{\n'
                )
                for kv in self.stack_update:
                    kv.serialise(file, start_indent='\t\t\t')
                file.write('\t\t\t}\n')
            if self.stack_stop:
                file.write(
                    '\t\t' 'stop_stack\n'
                    '\t\t\t' '{\n'
                )
                for kv in self.stack_stop:
                    kv.serialise(file, start_indent='\t\t\t')
                file.write('\t\t\t}\n')
            file.write('\t\t}\n')
        file.write('\t}\n')
