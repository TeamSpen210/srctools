"""Parses VCD choreo scenes, as well as data in scenes.image."""
from __future__ import annotations
from typing import ClassVar, List, IO, NewType, Dict, Optional, Tuple
from typing_extensions import Literal, TypeAlias

from io import BytesIO
import enum
import struct

import attrs

from srctools import binformat


CRC = NewType('CRC', int)
ScenesImage: TypeAlias = Dict[CRC, 'Entry']


def checksum_filename(filename: str) -> CRC:
    """Normalise the filename, then checksum it."""
    filename = filename.lower().replace('/', '\\')
    if not filename.startswith('scenes\\'):
        filename = 'scenes\\' + filename
    return CRC(binformat.checksum(filename.encode('ascii')))


def _update_checksum(choreo: Entry, attr: attrs.Attribute[str], value: str) -> None:
    """When set, the filename attribute automatically recalculates the checksum.

    If set to an empty string, the checksum is not changed since that indicates the name is not known.
    """
    if value:
        choreo.checksum = checksum_filename(value)


@attrs.define
class Entry:
    """An entry in ``scenes.image``, containing useful metadata about a scene as well as the scene itself."""
    #: The filename of the choreo scene. If parsed from scenes.image, only a CRC is available.
    #: When set, this automatically recalculates the checksum.
    filename: str = attrs.field(validator=_update_checksum)
    checksum: CRC  # CRC hash.
    duration_ms: int  # Duration in milliseconds.
    last_speak_ms: int  # Time at which the last voice line ends.
    sounds: List[str]  # List of sounds it uses.
    data: Scene

    @property
    def duration(self) -> float:
        """Return the duration in seconds."""
        return self.duration_ms / 1000.0

    @duration.setter
    def duration(self, value: float) -> None:
        """Set the duration (in seconds). This is rounded to the nearest millisecond."""
        self.duration_ms = round(value * 1000.0)

    @property
    def last_speak(self) -> float:
        """Return the last-speak time in seconds."""
        return self.last_speak_ms / 1000.0

    @last_speak.setter
    def last_speak(self, value: float) -> None:
        """Set the last-speak time (in seconds). This is rounded to the nearest millisecond."""
        self.last_speak_ms = round(value * 1000.0)


class Interpolation(enum.Enum):
    """Kinds of interpolation."""
    DEFAULT = 0
    CATMULL_ROM_NORMALIZE_X = 1
    EASE_IN = 2
    EASE_OUT = 3
    EASE_IN_OUT = 4
    BSP_LINE = 5
    LINEAR = 6
    KOCHANEK_BARTELS = 7
    KOCHANEK_BARTELS_EARLY = 8
    KOCHANEK_BARTELS_LATE = 9
    SIMPLE_CUBIC = 10
    CATMULL_ROM = 11
    CATMULL_ROM_NORMALIZE = 12
    CATMULL_ROM_TANGENT = 13
    EXPONENTIAL_DECAY = 14
    HOLD = 15

    @classmethod
    def parse_pair(cls, value: int) -> Tuple[Interpolation, Interpolation]:
        """Parse two interpolation types, packed into a byte."""
        return cls((value >> 8) & 0xff), cls(value & 0xff)


class EventType(enum.Enum):
    """Kinds of events."""
    Unspecified = 0
    Section = 1
    Expression = 2
    LookAt = 3
    MoveTo = 4
    Speak = 5
    Gesture = 6
    Sequence = 7
    Face = 8
    FireTrigger = 9
    FlexAnimation = 10
    SubScene = 11
    Loop = 12
    Interrupt = 13
    StopPoint = 14
    PermitResponses = 15
    Generic = 16
    Camera = 17
    Script = 18


class EventFlags(enum.Flag):
    """Flags for an event."""
    Resume = 1 << 0
    LockBodyFacing = 1 << 1
    FixedLength = 1<<2
    Active = 1<<3
    ForceShortMovement = 1<<4
    PlayOverScript = 1 << 5


class CaptionType(enum.Enum):
    """Kind of closed captions."""
    Master = 0
    Slave = 1
    Disabled = 2


@attrs.define
class ExpressionSample:
    time: float
    value: float
    curve_type: Tuple[Interpolation, Interpolation] = (Interpolation.DEFAULT, Interpolation.DEFAULT)


@attrs.define
class Tag:
    """A tag labels a particular location in an event."""
    name: str
    value: float

    @classmethod
    def parse(cls, file: IO[bytes], string_pool: List[str], double: bool) -> List[Tag]:
        """Parse a list of tags from the file. If double is set, the value is 16-bit not 8-bit."""
        [tag_count] = file.read(1)
        tags = []
        divisor = 4096.0 if double else 255.0
        structure = struct.Struct('<hH' if double else '<hB')
        for _ in range(tag_count):
            [name_ind, value] = binformat.struct_read(structure, file)
            tags.append(Tag(string_pool[name_ind], value / divisor))
        return tags


@attrs.define
class Curve:
    """Scene or event ramp data."""
    BIN_FMT: ClassVar[struct.Struct] = struct.Struct('<fB')

    ramp: List[ExpressionSample]
    # start, end = ramp edge info

    @classmethod
    def parse_binary(cls, file: IO[bytes]) -> Curve:
        """Parse the BVCD form of this data."""
        [count] = file.read(1)
        ramp = []
        for _ in range(count):
            [time, value] = binformat.struct_read(cls.BIN_FMT, file)
            ramp.append(ExpressionSample(time, value / 255.0))
        return cls(ramp)


# Using a Literal here means Event.__init__() doesn't allow Loop/Speak/Gesture as the type,
# but the attribute does allow those as results meaning the subclasses are still valid.
def _validate_base_event_type(value: Literal[
    EventType.Unspecified,
    EventType.Section,
    EventType.Expression,
    EventType.LookAt,
    EventType.MoveTo,
    EventType.Sequence,
    EventType.Face,
    EventType.FireTrigger,
    EventType.FlexAnimation,
    EventType.SubScene,
    EventType.Interrupt,
    EventType.StopPoint,
    EventType.PermitResponses,
    EventType.Generic,
    EventType.Camera,
    EventType.Script,
]) -> EventType:
    """Validate event types that can be passed to the base Event class.

    We don't allow those that require additional attributes (and therefore a subclass).
    """
    if value.name in {'Loop', 'Speak', 'Gesture'}:
        raise ValueError(
            'Event() must not be instantiated with '
            f'event type {value}, use {value.name}Event() instead.'
        )
    return value


@attrs.define(eq=False, kw_only=True)
class Event:
    """An event is an action that occurs in a choreo scene's timeline."""
    type: EventType = attrs.field(converter=_validate_base_event_type)
    name: str
    parameters: tuple[str, str, str]
    start_time: float
    end_time: float

    ramp: Curve
    tag_name: Optional[str] = None
    tag_wav_name: Optional[str] = None

    relative_tags: List[Tag] = attrs.Factory(list)
    timing_tags: List[Tag] = attrs.Factory(list)
    absolute_playback_tags: List[Tag] = attrs.Factory(list)
    absolute_original_tags: List[Tag] = attrs.Factory(list)
    flex_anim_tracks: List[FlexAnimTrack] = attrs.Factory(list)
    sub_scene: str = ''

    dist_to_targ: float = 0
    default_curve_type: Tuple[Interpolation, Interpolation] = (Interpolation.DEFAULT, Interpolation.DEFAULT)

    flags: EventFlags = EventFlags(0)

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Event:
        """Parse the BVCD form of this data."""
        [
            type_int, name_ind, start_time, end_time,
            param_ind1, param_ind2, param_ind3,
        ] = binformat.struct_read('<bhffhhh', file)
        event_type = EventType(type_int)
        parameters = (string_pool[param_ind1], string_pool[param_ind2], string_pool[param_ind3])
        ramp = Curve.parse_binary(file)
        [flags, dist_to_targ] = binformat.struct_read('<Bf', file)

        rel_tags = Tag.parse(file, string_pool, False)
        timing_tags = Tag.parse(file, string_pool, False)
        abs_playback_tags = Tag.parse(file, string_pool, True)
        abs_orig_tags = Tag.parse(file, string_pool, True)

        if event_type is EventType.Gesture:
            [gesture_sequence_duration] = binformat.struct_read('<f', file)
        else:
            gesture_sequence_duration = 0.0  # Never used.

        tag_name: Optional[str]
        tag_wav_name: Optional[str]
        if file.read(1) != b'\x00':
            # Using a relative tag
            [tag_name_ind, wav_name_ind] = binformat.struct_read('<hh', file)
            tag_name = string_pool[tag_name_ind]
            tag_wav_name = string_pool[wav_name_ind]
        else:
            tag_name = tag_wav_name = None

        [flex_count] = file.read(1)
        assert flex_count == 0  # TODO

        if event_type is EventType.Gesture:
            return GestureEvent(
                name=string_pool[name_ind],
                start_time=start_time,
                end_time=end_time,
                parameters=parameters,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                absolute_playback_tags=abs_playback_tags,
                absolute_original_tags=abs_orig_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,

                gesture_sequence_duration=gesture_sequence_duration,
            )
        if event_type is EventType.Loop:
            [loop_count] = binformat.struct_read('b', file)

            return LoopEvent(
                name=string_pool[name_ind],
                start_time=start_time,
                end_time=end_time,
                parameters=parameters,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                absolute_playback_tags=abs_playback_tags,
                absolute_original_tags=abs_orig_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,

                loop_count=loop_count,
            )
        elif event_type is EventType.Speak:
            [cc_type_ind, cc_token_ind, speak_flags] = binformat.struct_read('<Bhb', file)

            return SpeakEvent(
                name=string_pool[name_ind],
                start_time=start_time,
                end_time=end_time,
                parameters=parameters,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                absolute_playback_tags=abs_playback_tags,
                absolute_original_tags=abs_orig_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,

                caption_type=CaptionType(cc_type_ind),
                cc_token=string_pool[cc_token_ind],
                use_combined_file=(speak_flags & 1) != 0,
                use_gender_token=(speak_flags & 2) != 0,
                suppress_caption_attenuation=(speak_flags & 4) != 0,
            )
        else:
            return Event(
                type=event_type,
                name=string_pool[name_ind],
                start_time=start_time,
                end_time=end_time,
                parameters=parameters,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                absolute_playback_tags=abs_playback_tags,
                absolute_original_tags=abs_orig_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,
            )


@attrs.define(eq=False, kw_only=True)
class GestureEvent(Event):
    """Additional parameters for Gesture events."""
    type: Literal[EventType.Gesture] = attrs.field(default=EventType.Gesture, init=False, repr=False)
    gesture_sequence_duration: float


@attrs.define(eq=False, kw_only=True)
class LoopEvent(Event):
    """Additional parameters for Loop events."""
    type: Literal[EventType.Loop] = attrs.field(default=EventType.Loop, init=False, repr=False)
    loop_count: int = 0


@attrs.define(eq=False, kw_only=True)
class SpeakEvent(Event):
    """Additional parameters for Speak events."""
    type: Literal[EventType.Speak] = attrs.field(default=EventType.Speak, init=False, repr=False)
    caption_type: CaptionType = CaptionType.Master
    cc_token: str = ''
    suppress_caption_attenuation: bool = False
    use_combined_file: bool = False
    use_gender_token: bool = False


@attrs.define(eq=False)
class Channel:
    name: str
    active: bool = True
    events: List[Event] = attrs.Factory(list)

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Channel:
        """Parse the BVCD form of this data."""
        [name_ind, event_count] = binformat.struct_read('<hB', file)
        name = string_pool[name_ind]
        events = []
        for _ in range(event_count):
            events.append(Event.parse_binary(file, string_pool))
        active = file.read(1) != b'\x00'
        return cls(name, active, events)


@attrs.define(eq=False)
class Actor:
    name: str
    active: bool = True
    channels: List[Channel] = attrs.Factory(list)

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Actor:
        """Parse the BVCD form of this data."""
        [name_ind, channel_count] = binformat.struct_read('<hB', file)
        name = string_pool[name_ind]
        channels = []
        for _ in range(channel_count):
            channels.append(Channel.parse_binary(file, string_pool))
        active = file.read(1) != b'\x00'
        return cls(name, active, channels)


@attrs.define(eq=False, kw_only=True)
class Scene:
    """A choreo scene."""
    events: List[Event] = attrs.Factory(list)
    actors: List[Actor] = attrs.Factory(list)
    ramp: Curve = attrs.Factory(lambda: Curve([]))
    ignore_phonemes: bool = False

    # VCD only?
    # channels: List[Channel] = attrs.Factory(list)
    # map_name: str = ''
    # fps: int = 0
    # time_zoom_lookup: Dict[int, int] = attrs.Factory(dict)
    # is_background: bool = False
    # is_sub_scene: bool = False
    # use_frame_snap: bool = False

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Scene:
        """Parse the BVCD form of this data."""
        if file.read(4) != b'bvcd':
            raise ValueError('File is not a binary VCD scene!')
        version = file.read(1)[0]
        if version != 4:
            raise ValueError(f'Unknown version "{version}"!')
        [crc, event_count] = binformat.struct_read('<IB', file)

        events = [
            Event.parse_binary(file, string_pool)
            for _ in range(event_count)
        ]
        [actor_count] = file.read(1)
        actors = [
            Actor.parse_binary(file, string_pool)
            for _ in range(actor_count)
        ]
        ramp = Curve.parse_binary(file)
        ignore_phonemes = file.read(1) != b'\x00'

        return cls(
            events=events,
            actors=actors,
            ramp=ramp,
            ignore_phonemes=ignore_phonemes,
        )


def parse_scenes_image(file: IO[bytes]) -> ScenesImage:
    """Parse the ``scenes.image`` file, extracting all the choreo data."""
    [
        magic,
        version,
        scene_count,
        string_count,
        scene_off,
    ] = binformat.struct_read('<4s4i', file)
    if magic != b'VSIF':
        raise ValueError("Invalid scenes.image!")
    if version not in (2, 3):
        raise ValueError("Unknown version {}!".format(version))

    string_pool = binformat.read_offset_array(file, string_count)

    scenes: ScenesImage = {}

    file.seek(scene_off)
    scene_data: List[Tuple[CRC, int, int, int]] = [
        binformat.struct_read('<4i', file)
        for _ in range(scene_count)
    ]

    for (
        crc,
        data_off, data_size,
        summary_off,
    ) in scene_data:
        file.seek(summary_off)
        if version == 3:
            [duration, last_speak, sound_count] = binformat.struct_read('<3i', file)
        else:
            [duration, sound_count] = binformat.struct_read('<2i', file)
            last_speak = duration  # Assume it's the whole choreo scene.
        sounds = [
            string_pool[i]
            for i in binformat.struct_read('<{}i'.format(sound_count), file)
        ]
        file.seek(data_off)
        data = file.read(data_size)
        if data.startswith(b'LZMA'):
            data = binformat.decompress_lzma(data)
        scenes[crc] = Entry(
            '',
            crc,
            duration, last_speak,
            sounds,
            Scene.parse_binary(BytesIO(data), string_pool),
        )
    return scenes
