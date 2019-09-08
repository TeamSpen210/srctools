"""Reads and writes Soundscripts."""
from enum import Enum

from srctools import Property, conv_float

from typing import (
    Optional, Union, TypeVar,
    List, Tuple, Dict,
    Iterable, Type, Callable,
    TextIO,
)

# All the prefixes  wavs can have.
SND_CHARS = '*@#<>^)}$!?'


class Pitch(float, Enum):
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
    val: str, 
    enum: Callable[[str], EnumType], 
    default,
) -> Tuple[Union[float, EnumType], Union[float, EnumType]]:
    """Handle values which can be either single or a low, high pair of numbers.
    
    If single, low and high are the same.
    enum is a Enum with values to match text constants, or a converter function
    returning enums or raising ValueError, KeyError or IndexError.
    """
    if ',' in val:
        s_low, s_high = val.split(',')
        try: 
            low = enum(s_low.upper())
        except (LookupError, ValueError):
            low = conv_float(s_low, default)
        try: 
            high = enum(s_high.upper())
        except (LookupError, ValueError):
            high = conv_float(s_high, default)
        return low, high
    else:
        try: 
            out = enum(val.upper())
        except (LookupError, ValueError):
            out = conv_float(val, default)
        return out, out


def join_float(val) -> str:
    """Reverse split_float()."""
    low, high = val
    if low == high:
        return str(low)
    else:
        return '{!s},{!s}'.format(low, high)


class Sound:
    """Represents a single sound in the list."""
    def __init__(
        self,
        name: str,
        sounds: List[str],
        volume: Union[Tuple[Union[float, VOLUME], Union[float, VOLUME]], float, VOLUME]=VOL_NORM,
        channel: Channel=Channel.DEFAULT,
        level: Union[Tuple[Union[float, Level], Union[float, Level]], float, Level]=Level.SNDLVL_NORM,
        pitch: Union[Tuple[Union[float, Pitch], Union[float, Pitch]], float, Pitch]=Pitch.PITCH_NORM,
        
        # Operator stacks
        stack_start: Optional[Property]=None,
        stack_update: Optional[Property]=None,
        stack_stop: Optional[Property]=None,
        use_v2: bool=False,
    ) -> None:
        """Create a soundscript."""
        self.name = name
        self.sounds = sounds
        self.channel = channel
        self.force_v2 = use_v2

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
        
        self.stack_start = Property('', []) if stack_start is None else stack_start
        self.stack_update = Property('', []) if stack_update is None else stack_update
        self.stack_stop = Property('', []) if stack_stop is None else stack_stop
       
    @classmethod 
    def parse(cls, file: Property) -> Dict[str, 'Sound']:
        """Parses a soundscript file.
        
        This returns a dict mapping casefolded names to Sounds.
        """
        sounds = {}
        for snd_prop in file:
            volume = split_float(
                snd_prop['volume', '1'],
                VOLUME,
                1.0,
            )
            pitch = split_float(
                snd_prop['pitch', '100'],
                Pitch.__getitem__,
                100.0,
            )
            
            level = split_float(
                snd_prop['soundlevel', 'SNDLVL_NORM'],
                Level.__getitem__,
                Level.SNDLVL_NORM,
            )
            
            # Either 1 "wave", or multiple in "rndwave".
            wavs = []  # type: List[str]
            for prop in snd_prop:
                if prop.name == 'wave':
                    wavs.append(prop.value)
                elif prop.name == 'rndwave':
                    for subprop in prop:
                        wavs.append(subprop.value)

            channel = Channel(snd_prop['channel', 'CHAN_AUTO'])
            
            sound_version = snd_prop.int('soundentry_version', 1)
            
            if 'operator_stacks' in snd_prop:
                if sound_version == 1:
                    raise ValueError(
                        'Operator stacks used with version '
                        'less than 2 in "{}"!'.format(snd_prop.real_name))
                
                start_stack, update_stack, stop_stack = [
                    Property(stack_name, [
                        prop.copy()
                        for prop in 
                        snd_prop.find_children('operator_stacks', stack_name)
                    ])
                    for stack_name in 
                    ['start_stack', 'update_stack', 'stop_stack']
                ]
            else:
                start_stack, update_stack, stop_stack = [
                    Property(stack_name, [])
                    for stack_name in 
                    ['start_stack', 'update_stack', 'stop_stack']
                ]
            
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

    def export(self, file: TextIO):
        """Write a sound to a file.
        
        Pass a file-like object open for text writing.
        """
        file.write('"{}"\n\t{{\n'.format(self.name))

        file.write('\t' 'channel {}\n'.format(self.channel.value))

        file.write('\t' 'soundlevel {}\n'.format(join_float(self.level)))

        if self.volume != (1, 1):
            file.write('\tvolume {}\n'.format(join_float(self.volume)))
        if self.pitch != (100, 100):
            file.write('\tpitch {}\n'.format(join_float(self.pitch)))

        if len(self.sounds) > 1:
            file.write('\trndwave\n\t\t{\n')
            for wav in self.sounds:
                file.write('\t\twave "{}"\n'.format(wav))
            file.write('\t\t}\n')
        else:
            file.write('\twave "{}"\n'.format(self.sounds[0]))

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


