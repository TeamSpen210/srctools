"""Reads and writes Soundscripts."""
from typing import IO, Callable, Dict, List, Optional, TextIO, Tuple, TypeVar, Union
from enum import Enum
import struct

import attrs

from srctools import conv_float
from srctools.keyvalues import Keyvalues, NoKeyError


__all__ = [
    'SND_CHARS', 'Pitch', 'VOL_NORM', 'Channel', 'Level',
    'Sound', 'wav_is_looped',
]

# All the prefixes wavs can have.
SND_CHARS = '*@#<>^)}$!?'


class Pitch(float, Enum):
    """The constants permitted for sound pitches."""
    PITCH_NORM = 100.0
    PITCH_LOW = 95.0
    PITCH_HIGH = 120.0

    def __str__(self) -> str:
        return self.name


class VOLUME(Enum):
    """Special value, substitutes default volume (usually 1)."""
    VOL_NORM = 'VOL_NORM'

    def __str__(self) -> str:
        return self.name

VOL_NORM = VOLUME.VOL_NORM

# Old compatibility values, replaced by soundlevel.
ATTENUATION: Dict[str, float] = {
    'ATTN_NONE': 0.0,
    'ATTN_NORM': 0.8,
    'ATTN_IDLE': 2.0,
    'ATTN_STATIC': 1.25,
    'ATTN_RICOCHET': 1.5,
    'ATTN_GUNFIRE': 0.27,
}


class Channel(Enum):
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

    #CHAN_USER_BASE+<number>
    #Custom channels can be defined here.

    def __str__(self) -> str:
        return self.value


class Level(Enum):
    """Soundlevel constants - attenuation."""
    SNDLVL_NONE = 'SNDLVL_NONE'
    SNDLVL_20dB = 'SNDLVL_20dB'
    SNDLVL_25dB = 'SNDLVL_25dB'
    SNDLVL_30dB = 'SNDLVL_30dB'
    SNDLVL_35dB = 'SNDLVL_35dB'
    SNDLVL_40dB = 'SNDLVL_40dB'
    SNDLVL_45dB = 'SNDLVL_45dB'
    SNDLVL_50dB = 'SNDLVL_50dB'
    SNDLVL_55dB = 'SNDLVL_55dB'
    SNDLVL_IDLE = 'SNDLVL_IDLE'
    SNDLVL_65dB = 'SNDLVL_65dB'
    SNDLVL_STATIC = 'SNDLVL_STATIC'
    SNDLVL_70dB = 'SNDLVL_70dB'
    SNDLVL_NORM = 'SNDLVL_NORM'
    SNDLVL_80dB = 'SNDLVL_80dB'
    SNDLVL_TALKING = 'SNDLVL_TALKING'
    SNDLVL_85dB = 'SNDLVL_85dB'
    SNDLVL_90dB = 'SNDLVL_90dB'
    SNDLVL_95dB = 'SNDLVL_95dB'
    SNDLVL_100dB = 'SNDLVL_100dB'
    SNDLVL_105dB = 'SNDLVL_105dB'
    SNDLVL_110dB = 'SNDLVL_110dB'
    SNDLVL_120dB = 'SNDLVL_120dB'
    SNDLVL_125dB = 'SNDLVL_125dB'
    SNDLVL_130dB = 'SNDLVL_130dB'
    SNDLVL_GUNFIRE = 'SNDLVL_GUNFIRE'
    SNDLVL_140dB = 'SNDLVL_140dB'
    SNDLVL_145dB = 'SNDLVL_145dB'
    SNDLVL_150dB = 'SNDLVL_150dB'
    SNDLVL_180dB = 'SNDLVL_180dB'

    def __str__(self) -> str:
        return self.name


EnumType = TypeVar('EnumType', bound=Enum)


def split_float(
    kv: Keyvalues,
    key: str,
    enum: Callable[[str], Union[float, EnumType]],
    default: Union[float, EnumType],
) -> Tuple[Union[float, EnumType], Union[float, EnumType]]:
    """Handle values which can be a low, high pair of numbers or enum constants.

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
    val = leaf_kv.value
    if ',' in val:
        s_low, s_high = val.split(',')
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
            out = enum(val.strip().upper())
        except (LookupError, ValueError):
            out = conv_float(val, default)
        return out, out


def join_float(val: Tuple[Union[float, Enum], Union[float, Enum]]) -> str:
    """Reverse split_float(). The two parameters should be stringifiable into floats/constants."""
    low, high = val
    if low == high:
        return str(low)
    else:
        return f'{low!s}, {high!s}'


class _WAVChunk:
    """To allow reading CUE points, we need a copy of the former chunk module.

    This is copied from the Python Standard Library, 3.11.
    We force little-endian, and to align to 2-byte boundaries.
    """
    def __init__(self, file: IO[bytes]) -> None:
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
        try:
            self.offset = self.file.tell()
        except (AttributeError, OSError):
            self.seekable = False
        else:
            self.seekable = True

    def close(self) -> None:
        """Close this chunk, skipping to the next."""
        if not self.closed:
            try:
                self.skip()
            finally:
                self.closed = True

    def seek(self, pos: int, whence: int = 0) -> None:
        """Seek to specified position into the chunk.
        Default position is 0 (start of chunk).
        If the file is not seekable, this will result in an error.
        """

        if self.closed:
            raise ValueError("I/O operation on closed file")
        if not self.seekable:
            raise OSError("cannot seek")
        if whence == 1:
            pos = pos + self.size_read
        elif whence == 2:
            pos = pos + self.chunksize
        if pos < 0 or pos > self.chunksize:
            raise RuntimeError
        self.file.seek(self.offset + pos, 0)
        self.size_read = pos

    def read(self, size: int = -1) -> bytes:
        """Read at most size bytes from the chunk.
        If size is omitted or negative, read until the end
        of the chunk.
        """
        if self.closed:
            raise ValueError("I/O operation on closed file")
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
        if self.closed:
            raise ValueError("I/O operation on closed file")
        if self.seekable:
            try:
                n = self.chunksize - self.size_read
                # maybe fix alignment
                if self.chunksize & 1:
                    n = n + 1
                self.file.seek(n, 1)
                self.size_read = self.size_read + n
                return
            except OSError:
                pass
        while self.size_read < self.chunksize:
            n = min(8192, self.chunksize - self.size_read)
            dummy = self.read(n)
            if not dummy:
                raise EOFError


def wav_is_looped(file: IO[bytes]) -> bool:
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
    sounds: List[str] = attrs.Factory(list)
    volume: Tuple[Union[float, VOLUME], Union[float, VOLUME]] = (VOL_NORM, VOL_NORM)
    channel: Union[int, Channel] = Channel.DEFAULT
    level: Tuple[Union[float, Level], Union[float, Level]] = (Level.SNDLVL_NORM, Level.SNDLVL_NORM)
    pitch: Tuple[Union[float, Pitch], Union[float, Pitch]] = (Pitch.PITCH_NORM, Pitch.PITCH_NORM)

    _stack_start: Optional[Keyvalues] = None
    _stack_update: Optional[Keyvalues] = None
    _stack_stop: Optional[Keyvalues] = None
    force_v2: bool = False

    def __init__(
        self,
        name: str,
        sounds: List[str],
        volume: Union[Tuple[Union[float, VOLUME], Union[float, VOLUME]], float, VOLUME]=(VOL_NORM, VOL_NORM),
        channel: Union[int, Channel]=Channel.DEFAULT,
        level: Union[Tuple[Union[float, Level], Union[float, Level]], float, Level]=(Level.SNDLVL_NORM, Level.SNDLVL_NORM),
        pitch: Union[Tuple[Union[float, Pitch], Union[float, Pitch]], float, Pitch]=(Pitch.PITCH_NORM, Pitch.PITCH_NORM),

        # Operator stacks
        stack_start: Optional[Keyvalues]=None,
        stack_update: Optional[Keyvalues]=None,
        stack_stop: Optional[Keyvalues]=None,
        force_v2: bool=False,
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
    def parse(cls, file: Keyvalues) -> Dict[str, 'Sound']:
        """Parses a soundscript file.

        This returns a dict mapping casefolded names to Sounds.
        """
        sounds = {}
        for snd_prop in file:
            volume = split_float(
                snd_prop, 'volume',
                VOLUME,
                1.0,
            )
            pitch = split_float(
                snd_prop, 'pitch',
                Pitch.__getitem__,
                100.0,
            )

            if 'soundlevel' in snd_prop:
                level = split_float(
                    snd_prop, 'soundlevel',
                    Level.__getitem__,
                    Level.SNDLVL_NORM,
                )
            elif 'attenuation' in snd_prop:
                atten_min, atten_max = split_float(
                    snd_prop, 'attenuation',
                    ATTENUATION.__getitem__,
                    ATTENUATION['ATTN_IDLE'],
                )
                # Convert to a soundlevel.
                # See source_sdk/public/soundflags.h:ATTN_TO_SNDLVL()
                level = (
                    (50.0 + 20.0 / atten_min) if atten_min else 0.0,
                    (50.0 + 20.0 / atten_max) if atten_max else 0.0,
                )
            else:
                level = (Level.SNDLVL_NORM, Level.SNDLVL_NORM)

            # Either 1 "wave", or multiple in "rndwave".
            wavs: List[str] = []
            for prop in snd_prop:
                if prop.name == 'wave':
                    wavs.append(prop.value)
                elif prop.name == 'rndwave':
                    for subprop in prop:
                        wavs.append(subprop.value)

            channel_str = snd_prop['channel', 'CHAN_AUTO'].upper()
            channel: Union[int, Channel]
            if channel_str.startswith('CHAN_'):
                channel = Channel(channel_str)
            else:
                channel = int(channel_str)

            sound_version = snd_prop.int('soundentry_version', 1)

            if 'operator_stacks' in snd_prop:
                if sound_version == 1:
                    raise ValueError(
                        'Operator stacks used with version '
                        'less than 2 in "{}"!'.format(snd_prop.real_name))

                start_stack, update_stack, stop_stack = [
                    Keyvalues(stack_name, [
                        prop.copy()
                        for prop in
                        snd_prop.find_children('operator_stacks', stack_name)
                    ])
                    for stack_name in
                    ['start_stack', 'update_stack', 'stop_stack']
                ]
            else:
                start_stack, update_stack, stop_stack = [None, None, None]

            sounds[snd_prop.name] = Sound(
                snd_prop.real_name,
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
        return sounds

    def export(self, file: TextIO) -> None:
        """Write a sound to a file.

        Pass a file-like object open for text writing.
        """
        file.write(f'"{self.name}"\n\t{{\n')
        file.write(f'\tchannel {self.channel}\n')
        file.write(f'\tsoundlevel {join_float(self.level)}\n')

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
                for prop in self.stack_start:
                    for line in prop.export():
                        file.write('\t\t\t' + line)
                file.write('\t\t\t}\n')
            if self.stack_update:
                file.write(
                    '\t\t' 'update_stack\n'
                    '\t\t\t' '{\n'
                )
                for prop in self.stack_update:
                    for line in prop.export():
                        file.write('\t\t\t' + line)
                file.write('\t\t\t}\n')
            if self.stack_stop:
                file.write(
                    '\t\t' 'stop_stack\n'
                    '\t\t\t' '{\n'
                )
                for prop in self.stack_stop:
                    for line in prop.export():
                        file.write('\t\t\t' + line)
                file.write('\t\t\t}\n')
            file.write('\t\t}\n')
        file.write('\t}\n')
