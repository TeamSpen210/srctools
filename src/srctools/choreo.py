"""Parses VCD choreo scenes, as well as data in scenes.image."""
from __future__ import annotations

from typing import (
    Callable, ClassVar, Final, Iterable, List, IO, NewType, Dict, Optional, Tuple,
    Union,
)
from typing_extensions import Literal, Self, TypeAlias

from io import BytesIO
import enum
import struct
import re

import attrs

from srctools import binformat, conv_bool, conv_int, conv_float
from srctools.tokenizer import BaseTokenizer, Token, escape_text


CRC = NewType('CRC', int)
ScenesImage: TypeAlias = Dict[CRC, 'Entry']
# Only known binary version.
BINARY_VERSION: Final = 4


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


@attrs.define(eq=False)
class Entry:
    """An entry in ``scenes.image``, containing useful metadata about a scene.

    The data attribute may be accessed to parse the scene.
    """
    #: The filename of the choreo scene. If parsed from scenes.image, only a CRC is available.
    #: When set, this automatically recalculates the checksum.
    filename: str = attrs.field(validator=_update_checksum)
    checksum: CRC  # CRC hash.
    duration_ms: int  # Duration in milliseconds.
    last_speak_ms: int  # Time at which the last voice line ends.
    sounds: List[str]  # List of sounds it uses.
    # Either an already parsed scene, or the raw bytes plus the whole string pool.
    _data: Union[Scene, Tuple[bytes, List[str]]] = attrs.field(repr=False)

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

    @property
    def data(self) -> Scene:
        """The scene for this entry. When accessed this will parse the data if required."""
        if isinstance(self._data, tuple):
            data, string_pool = self._data
            self._data = Scene.parse_binary(BytesIO(data), string_pool)
        return self._data

    @data.setter
    def data(self, scene: Scene) -> None:
        self._data = scene


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

INTERP_TO_NAME = {
    Interpolation.DEFAULT: 'default',
    Interpolation.CATMULL_ROM_NORMALIZE_X: 'catmullrom_normalize_x',
    Interpolation.EASE_IN: 'easein',
    Interpolation.EASE_OUT: 'easeout',
    Interpolation.EASE_IN_OUT: 'easeinout',
    Interpolation.BSP_LINE: 'bspline',
    Interpolation.LINEAR: 'linear_interp',
    Interpolation.KOCHANEK_BARTELS: 'kochanek',
    Interpolation.KOCHANEK_BARTELS_EARLY: 'kochanek_early',
    Interpolation.KOCHANEK_BARTELS_LATE: 'kochanek_late',
    Interpolation.SIMPLE_CUBIC: 'simple_cubic',
    Interpolation.CATMULL_ROM: 'catmullrom',
    Interpolation.CATMULL_ROM_NORMALIZE: 'catmullrom_normalize',
    Interpolation.CATMULL_ROM_TANGENT: 'catmullrom_tangent',
    Interpolation.EXPONENTIAL_DECAY: 'exponential_decay',
    Interpolation.HOLD: 'hold',
}
NAME_TO_INTERP = {v: k for k, v in INTERP_TO_NAME.items()}


@attrs.frozen
class CurveType:
    """A pair of interpolation types."""
    first: Interpolation
    second: Interpolation

    @classmethod
    def parse_text(cls, text: str) -> CurveType:
        """Parse text in the form 'curve_AAA_to_curve_BBB'."""
        match = re.match('curve_([a-z_]+)_to_curve_([a-z_]+)', text.casefold())
        if match is None:
            raise ValueError('Invalid curve type text.')
        left, right = match.groups()
        return cls(NAME_TO_INTERP[left], NAME_TO_INTERP[right])

    @classmethod
    def parse_binary(cls, value: int) -> CurveType:
        """Parse two interpolation types, packed into a two-byte value."""
        return cls(Interpolation((value >> 8) & 0xff), Interpolation(value & 0xff))

    def export_binary(self) -> int:
        """Return the two interpolation types packed into a two-byte value."""
        return (self.first.value << 8) | self.second.value

    def __str__(self) -> str:
        """Return the associated name for this pair."""
        return f'curve_{INTERP_TO_NAME[self.first]}_to_curve_{INTERP_TO_NAME[self.second]}'


CURVE_DEFAULT = CurveType(Interpolation.DEFAULT, Interpolation.DEFAULT)


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
    ResumeCondition = 1 << 0
    LockBodyFacing = 1 << 1
    FixedLength = 1<<2
    Active = 1<<3
    ForceShortMovement = 1<<4
    PlayOverScript = 1 << 5


NAME_TO_EVENT_FLAG = {
    'resumecondition': EventFlags.ResumeCondition,
    'lockbodyfacing': EventFlags.LockBodyFacing,
    'fixedlength': EventFlags.FixedLength,
    'forceshortmovement': EventFlags.ForceShortMovement,
    'playoverscript': EventFlags.PlayOverScript,
    # Active is not included, works differently.
}
NAME_TO_EVENT_TYPE = {
    event.name.casefold(): event
    for event in EventType
}
# Match to the index in our list for easier parsing.
PARAM_KEY_INDEXES = {
    'param': 0,
    'param2': 1,
    'paraam3': 2,
}


class CaptionType(enum.Enum):
    """Kind of closed captions."""
    Master = 0
    Slave = 1
    Disabled = 2


@attrs.define
class ExpressionSample:
    """Keyframes for animations."""
    time: float
    value: float
    curve_type: CurveType = CURVE_DEFAULT


@attrs.define
class Tag:
    """A tag labels a particular location in an event."""
    _FMT: ClassVar[struct.Struct] = struct.Struct('<hB')
    _FACTOR: ClassVar[float] = 255.0
    _MAX: ClassVar[int] = 255

    name: str
    value: float = attrs.field(validator=[attrs.validators.ge(0.0), attrs.validators.le(1.0)])

    @classmethod
    def parse(cls, file: IO[bytes], string_pool: List[str], double: bool) -> List[Self]:
        """Parse a list of tags from the file. If double is set, the value is 16-bit not 8-bit."""
        [tag_count] = file.read(1)
        tags = []
        for _ in range(tag_count):
            [name_ind, value] = binformat.struct_read(cls._FMT, file)
            tags.append(cls(string_pool[name_ind], value / cls._FACTOR))
        return tags

    @classmethod
    def export_binary(cls, file: IO[bytes], add_to_pool: Callable[[str], int], tags: List[Self]) -> None:
        """Write this to a binary BVCD block."""
        file.write(struct.pack('B', len(tags)))
        for tag in tags:
            value = min(cls._MAX, max(0, round(tag.value * cls._FACTOR)))
            file.write(cls._FMT.pack(add_to_pool(tag.name), value))
            # Timing tags do not save the locked state.

    @classmethod
    def export_text(cls, file: IO[str], indent: str, tags: List[Self], block_name: str) -> None:
        """Export a list of tags into a text VCD file."""
        if not tags:
            return
        file.write(f'{indent} {block_name}\n{indent}  {{\n')
        for tag in tags:
            if isinstance(tag, TimingTag):
                lock = ' 1' if tag.locked else ' 0'
            else:
                lock = ''
            file.write(f'{indent}  "{escape_text(tag.name)}" {tag.value}{lock}\n')
        file.write(f'{indent}  }}\n')


@attrs.define
class TimingTag(Tag):
    """Flex animation timing tags additionally can be locked."""
    # VCD only.
    locked: bool = False


class AbsoluteTag(Tag):
    """Absolute tags have an increased range and precision."""
    _FMT: ClassVar[struct.Struct] = struct.Struct('<hH')
    _FACTOR: ClassVar[float] = 4096.0
    _MAX: ClassVar[int] = 65535

    value: float = attrs.field(validator=[attrs.validators.ge(0.0), attrs.validators.lt(16.0)])


@attrs.frozen
class CurveEdge:
    """Curve data, only saved in the text file."""
    active: bool
    zero_pos: float = 0.0
    curve_type: CurveType = CURVE_DEFAULT


@attrs.define
class Curve:
    """Scene or event ramp data."""
    BIN_FMT: ClassVar[struct.Struct] = struct.Struct('<fB')

    ramp: List[ExpressionSample] = attrs.Factory(list)
    # VCD only
    left: CurveEdge = CurveEdge(False)
    right: CurveEdge = CurveEdge(False)

    @classmethod
    def parse_binary(cls, file: IO[bytes]) -> Self:
        """Parse the BVCD form of this data."""
        [count] = file.read(1)
        ramp = []
        for _ in range(count):
            [time, value] = binformat.struct_read(cls.BIN_FMT, file)
            ramp.append(ExpressionSample(time, value / 255.0))
        return cls(ramp)

    def export_binary(self, file: IO[bytes]) -> None:
        """Write this to a binary BVCD block."""
        file.write(struct.pack('B', len(self.ramp)))
        for sample in self.ramp:
            value = min(255, max(0, round(sample.value * 255.0)))
            file.write(self.BIN_FMT.pack(sample.time, value))

    def export_text(self, file: IO[str], indent: str, name: str) -> None:
        """Write this to a text VCD file."""
        if not self.ramp and not self.left.active and not self.right.active:
            return
        file.write(f'{indent} {name}')
        if self.left.active:
            file.write(f' leftedge {self.left.curve_type} {self.left.zero_pos}')
        if self.right.active:
            file.write(f' rightedge {self.right.curve_type} {self.right.zero_pos}')
        file.write(f'\n{indent}  {{\n')
        for sample in self.ramp:
            curve = f' "{sample.curve_type}"' if sample.curve_type != CURVE_DEFAULT else ''
            file.write(f'{indent}  {sample.time} {sample.value}{curve}\n')
        file.write(f'{indent}  }}\n')


@attrs.define
class FlexAnimTrack:
    """Flex controller animation data."""
    name: str
    active: bool = True
    min: float = 0.0
    max: float = 1.0
    mag_track: List[ExpressionSample] = attrs.Factory(list)
    dir_track: Optional[List[ExpressionSample]] = None

    # VCD only
    left: CurveEdge = CurveEdge(False)
    right: CurveEdge = CurveEdge(False)

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> FlexAnimTrack:
        """Parse the BVCD form of this data."""
        [name_ind, flags, mins, maxes, track_count] = binformat.struct_read('<hBffh', file)
        active = flags & 1 != 0
        has_direction = flags & 2 != 0
        mag_track = []
        for _ in range(track_count):
            [time, value, curve_type] = binformat.struct_read('<fBH', file)
            mag_track.append(ExpressionSample(
                time, value / 255.0,
                CurveType.parse_binary(curve_type),
            ))

        if has_direction:
            dir_track = []
            [track_count] = binformat.struct_read('<H', file)
            for _ in range(track_count):
                [time, value, curve_type] = binformat.struct_read('<fBH', file)
                dir_track.append(ExpressionSample(
                    time, value / 255.0,
                    CurveType.parse_binary(curve_type),
                ))
        else:
            dir_track = None
        return cls(
            name=string_pool[name_ind],
            active=active,
            min=mins,
            max=maxes,
            mag_track=mag_track,
            dir_track=dir_track,
        )

    def export_binary(self, file: IO[bytes], add_to_pool: Callable[[str], int]) -> None:
        """Write this to a binary BVCD block."""
        flags = 1 * self.active | 2 * (self.dir_track is not None)
        file.write(struct.pack(
            '<hBffh',
            add_to_pool(self.name),
            flags,
            self.min, self.max,
            len(self.mag_track),
        ))
        for track in self.mag_track:
            value = min(255, max(0, round(track.value * 255.0)))
            file.write(struct.pack(
                '<fBh',
                track.time, track.value,
                track.curve_type.export_binary(),
            ))
        if self.dir_track is not None:
            file.write(struct.pack('<H', len(self.dir_track)))
            for track in self.dir_track:
                value = min(255, max(0, round(track.value * 255.0)))
                file.write(struct.pack(
                    '<fBh',
                    track.time, track.value,
                    track.curve_type.export_binary(),
                ))

    def export_text(self, file: IO[str], indent: str, default_curve: CurveType) -> None:
        """Write this to a text VCD file."""
        file.write(f'{indent}  "{escape_text(self.name)}"')
        if not self.active:
            file.write(' disabled')
        if self.dir_track is not None:
            file.write(' combo')
        if self.min != 0.0 or self.max != 1.0:
            file.write(f' range {self.min} {self.max}')
        if self.left.active:
            file.write(f' leftedge {self.left.curve_type} {self.left.zero_pos}')
        if self.right.active:
            file.write(f' rightedge {self.right.curve_type} {self.right.zero_pos}')
        file.write(f'\n{indent}   {{\n')
        for sample in self.mag_track:
            curve = f' "{sample.curve_type}"' if sample.curve_type != default_curve else ''
            file.write(f'{indent}   {sample.time} {sample.value}{curve}\n')
        file.write(f'{indent}   }}\n')
        if self.dir_track is not None:
            file.write(f'{indent}   {{\n')
            for sample in self.dir_track:
                curve = f' "{sample.curve_type}"' if sample.curve_type != default_curve else ''
                file.write(f'{indent}   {sample.time} {sample.value}{curve}\n')
            file.write(f'{indent}  }}\n')


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


def _check_event_type(
    tokenizer: BaseTokenizer, key: str,
    event: EventType, expected: EventType,
) -> None:
    """Raise an error if the event types do not match."""
    if event is not expected:
        raise tokenizer.error(
            'Only {} events can use {}, not {}!',
            EventType.Gesture.name, key, event.name,
        )


@attrs.define(eq=False, kw_only=True)
class Event:
    """An event is an action that occurs in a choreo scene's timeline."""
    name: str
    type: EventType = attrs.field(converter=_validate_base_event_type)
    flags: EventFlags = EventFlags(0)
    parameters: tuple[str, str, str]
    start_time: float
    end_time: float

    ramp: Curve
    tag_name: Optional[str] = None
    tag_wav_name: Optional[str] = None
    dist_to_targ: float = 0

    relative_tags: List[Tag] = attrs.Factory(list)
    timing_tags: List[TimingTag] = attrs.Factory(list)
    absolute_playback_tags: List[AbsoluteTag] = attrs.Factory(list)
    absolute_shifted_tags: List[AbsoluteTag] = attrs.Factory(list)
    flex_anim_tracks: List[FlexAnimTrack] = attrs.Factory(list)

    # Only used in VCDs.
    default_curve_type: CurveType = CURVE_DEFAULT
    pitch: int = 0
    yaw: int = 0

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
        timing_tags = TimingTag.parse(file, string_pool, False)
        abs_playback_tags = AbsoluteTag.parse(file, string_pool, True)
        abs_shifted_tags = AbsoluteTag.parse(file, string_pool, True)

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
        flex_anims = [
            FlexAnimTrack.parse_binary(file, string_pool)
            for _ in range(flex_count)
        ]

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
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
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
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
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
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
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
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,
            )

    def export_binary(self, file: IO[bytes], add_to_pool: Callable[[str], int]) -> None:
        """Write this to a binary BVCD block."""
        file.write(struct.pack(
            '<bhffhhh',
            self.type.value,
            add_to_pool(self.name),
            self.start_time, self.end_time,
            add_to_pool(self.parameters[0]),
            add_to_pool(self.parameters[1]),
            add_to_pool(self.parameters[2]),
        ))
        self.ramp.export_binary(file)
        file.write(struct.pack('<Bf', self.flags.value, self.dist_to_targ))
        Tag.export_binary(file, add_to_pool, self.relative_tags)
        TimingTag.export_binary(file, add_to_pool, self.timing_tags)
        AbsoluteTag.export_binary(file, add_to_pool, self.absolute_playback_tags)
        AbsoluteTag.export_binary(file, add_to_pool, self.absolute_shifted_tags)
        if isinstance(self, GestureEvent):
            file.write(struct.pack('f', self.gesture_sequence_duration))
        if self.tag_name is not None or self.tag_wav_name is not None:
            file.write(b'\x01')
            file.write(struct.pack(
                '<Bhh', True,
                add_to_pool(self.tag_name or ''),
                add_to_pool(self.tag_wav_name or '')
            ))
        else:
            file.write(b'\x00')
        file.write(struct.pack('<B', len(self.flex_anim_tracks)))
        for track in self.flex_anim_tracks:
            track.export_binary(file, add_to_pool)
        if isinstance(self, LoopEvent):
            file.write(struct.pack('<b', self.loop_count))
        elif isinstance(self, SpeakEvent):
            flags = (
                1 * (self.caption_type is not CaptionType.Disabled and self.use_combined_file)
                | 2 * self.use_gender_token
                | 4 * self.suppress_caption_attenuation
            )
            file.write(struct.pack(
                '<bhb',
                self.caption_type.value,
                add_to_pool(self.cc_token),
                flags,
            ))

    @classmethod
    def parse_text(cls, tokenizer: BaseTokenizer) -> Event:
        """Parse text data. The 'event' string should have already been parsed."""
        event_type_str = tokenizer.expect(Token.STRING)
        name = tokenizer.expect(Token.STRING)
        try:
            event_type = NAME_TO_EVENT_TYPE[event_type_str.casefold()]
        except KeyError:
            raise tokenizer.error('Invalid event type "{}" for event "{}"!', event_type_str, name)

        tokenizer.expect(Token.BRACE_OPEN)
        flags = EventFlags.Active
        params = ['', '', '']
        start_time = 0.0
        end_time = -1.0

        ramp = Curve()
        tag_name: str | None = None
        tag_wav_name: str | None = None
        dist_to_targ: float = 0
        default_curve_type = CURVE_DEFAULT
        pitch = 0.0
        yaw = 0.0

        rel_tags: list[Tag] = []
        timing_tags: list[TimingTag] = []
        abs_playback_tags: list[AbsoluteTag] = []
        abs_shifted_tags: list[AbsoluteTag] = []
        flex_anims: list[FlexAnimTrack] = []

        gesture_sequence_duration = 0.0
        loop_count = -1

        caption_type = CaptionType.Master
        cc_token = ''
        suppress_caption_attenuation = False
        use_combined_file = False
        use_gender_token = False

        for tok, tok_str in tokenizer:
            if tok is Token.NEWLINE:
                continue
            if tok is Token.BRACE_CLOSE:
                break
            if tok is not Token.STRING:
                raise tokenizer.error(tok, tok_str)
            folded = tok_str.casefold()
            if folded == "time":
                start_str = tokenizer.expect(Token.STRING)
                end_str = tokenizer.expect(Token.STRING)
                try:
                    start_time = float(start_str)
                    end_time = float(end_str)
                except ValueError as exc:
                    raise tokenizer.error(
                        'Invalid duration values {} {}',
                        start_str, end_str,
                    ) from exc
            elif folded in PARAM_KEY_INDEXES:
                params[PARAM_KEY_INDEXES[folded]] = tokenizer.expect(Token.STRING, skip_newline=False)
            elif folded in NAME_TO_EVENT_FLAG:
                flags |= NAME_TO_EVENT_FLAG[folded]
            elif folded == "active":
                if conv_bool(tokenizer.expect(Token.STRING, skip_newline=False)):
                    flags |= EventFlags.Active
                else:
                    flags &= ~EventFlags.Active
            elif folded == "pitch":
                pitch = conv_int(tokenizer.expect(Token.STRING, skip_newline=False))
            elif folded == "yaw":
                yaw = conv_int(tokenizer.expect(Token.STRING, skip_newline=False))
            elif folded == "distancetotarget":
                dist_to_targ = conv_float(tokenizer.expect(Token.STRING, skip_newline=False))
            elif folded == "relativetag":
                tag_name = tokenizer.expect(Token.STRING, skip_newline=False)
                tag_wav_name = tokenizer.expect(Token.STRING, skip_newline=False)
            elif folded == "sequenceduration":
                _check_event_type(tokenizer, folded, event_type, EventType.Gesture)
                gesture_sequence_duration = conv_float(tokenizer.expect(Token.STRING, skip_newline=False))
            elif folded == "loopcount":
                _check_event_type(tokenizer, folded, event_type, EventType.Loop)
                loop_count = conv_int(tokenizer.expect(Token.STRING, skip_newline=False))
            elif folded == "cctype":
                _check_event_type(tokenizer, folded, event_type, EventType.Speak)
                cc_type_str = tokenizer.expect(Token.STRING, skip_newline=False)
                try:
                    caption_type = CaptionType(cc_type_str)
                except ValueError as exc:
                    raise tokenizer.error('Invalid caption type "{}"', cc_type_str) from exc
            elif folded == "cctoken":
                _check_event_type(tokenizer, folded, event_type, EventType.Speak)
                cc_token = tokenizer.expect(Token.STRING, skip_newline=False)
            elif folded == "cc_usingcombinedfile":
                _check_event_type(tokenizer, folded, event_type, EventType.Speak)
                use_combined_file = True
            elif folded == "cc_combinedusesgender":
                _check_event_type(tokenizer, folded, event_type, EventType.Speak)
                use_gender_token = True
            elif folded == "cc_noattenuate":
                _check_event_type(tokenizer, folded, event_type, EventType.Speak)
                suppress_caption_attenuation = True
            elif folded == "tags":
                raise NotImplementedError  # TODO
            elif folded == "absolutetags":
                raise NotImplementedError  # TODO
            elif folded == "flextimingtags":
                raise NotImplementedError  # TODO
            elif folded == "flexanimations":
                raise NotImplementedError  # TODO
            elif folded == "event_ramp":
                raise NotImplementedError  # TODO
            else:
                raise tokenizer.error('Unknown event keyvalue "{}"!', tok_str)
        else:
            raise tokenizer.error(Token.EOF)

        param_tup = tuple(params)
        assert len(param_tup) == 3

        # Pick the appropriate type.
        if event_type is EventType.Gesture:
            return GestureEvent(
                name=name,
                start_time=start_time,
                end_time=end_time,
                parameters=param_tup,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,

                gesture_sequence_duration=gesture_sequence_duration,
            )
        elif event_type is EventType.Loop:
            return LoopEvent(
                name=name,
                start_time=start_time,
                end_time=end_time,
                parameters=param_tup,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,

                loop_count=loop_count,
            )
        elif event_type is EventType.Speak:
            return SpeakEvent(
                name=name,
                start_time=start_time,
                end_time=end_time,
                parameters=param_tup,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,

                cc_token=cc_token,
                caption_type=caption_type,
                suppress_caption_attenuation=suppress_caption_attenuation,
                use_gender_token=use_gender_token,
                use_combined_file=use_combined_file,
            )
        else:
            return Event(
                type=event_type,
                name=name,
                start_time=start_time,
                end_time=end_time,
                parameters=param_tup,
                ramp=ramp,
                flags=EventFlags(flags),
                dist_to_targ=dist_to_targ,
                relative_tags=rel_tags,
                timing_tags=timing_tags,
                flex_anim_tracks=flex_anims,
                absolute_playback_tags=abs_playback_tags,
                absolute_shifted_tags=abs_shifted_tags,
                tag_name=tag_name,
                tag_wav_name=tag_wav_name,
            )

    def export_text(self, file: IO[str], indent: str) -> None:
        """Write this to a text VCD file."""
        file.write(f'{indent}event {self.type.name.lower()} "{escape_text(self.name)}"\n')
        file.write(f'{indent} {{\n')
        file.write(f'{indent} time {self.start_time} {self.end_time}\n')
        file.write(f'{indent} param "{escape_text(self.parameters[0])}"\n')
        if self.parameters[1]:
            file.write(f'{indent} param2 "{escape_text(self.parameters[1])}"\n')
        if self.parameters[2]:
            file.write(f'{indent} param3 "{escape_text(self.parameters[2])}"\n')
        if self.ramp.ramp:
            self.ramp.export_text(file, indent, 'event_ramp')
        if self.pitch:
            file.write(f'{indent} pitch {self.pitch}\n')
        if self.yaw:
            file.write(f'{indent} yaw {self.yaw}\n')
        if self.dist_to_targ > 0.0:
            file.write(f'{indent} distancetotarget {self.dist_to_targ:.2f}\n')
        for text, flag in NAME_TO_EVENT_FLAG.items():
            if flag in self.flags:
                file.write(f'{indent} {text}\n')
        # Active takes a bool to indicate if enabled/disabled. It starts enabled, so only write
        # if unset present.
        if EventFlags.Active not in self.flags:
            file.write(f'{indent} active 0\n')
        Tag.export_text(file, indent, self.relative_tags, 'tags')
        TimingTag.export_text(file, indent, self.timing_tags, 'flextimingtags')
        AbsoluteTag.export_text(file, indent, self.absolute_playback_tags, 'absolutetags playback_time')
        AbsoluteTag.export_text(file, indent, self.absolute_shifted_tags, 'absolutetags shifted_time')

        if isinstance(self, GestureEvent) and self.gesture_sequence_duration:
            file.write(f'{indent} sequenceduration {self.gesture_sequence_duration}\n')

        if self.tag_name is not None or self.tag_wav_name is not None:
            file.write(
                f'{indent} relativetag '
                f'"{escape_text(self.tag_name or "")}" '
                f'"{escape_text(self.tag_wav_name or "")}"\n'
            )
        if self.flex_anim_tracks:
            if self.default_curve_type != CURVE_DEFAULT:
                default_curve = self.default_curve_type
                curve = f' defaultcurvetype={self.default_curve_type}'
            else:
                default_curve = CURVE_DEFAULT
                curve = ''
            file.write(f'{indent} flexanimations samples_use_time{curve}\n{indent}  {{\n')
            for track in self.flex_anim_tracks:
                track.export_text(file, indent, default_curve)

        if isinstance(self, LoopEvent):
            file.write(f'{indent} loopcount "{self.loop_count}"\n')
        if isinstance(self, SpeakEvent):
            file.write(f'{indent} cctype "{self.caption_type.name.lower()}"\n')
            file.write(f'{indent} cctoken "{self.cc_token}"\n')
            if self.caption_type is not CaptionType.Disabled and self.use_combined_file:
                file.write(f'{indent} cc_usingcombinedfile\n')
            if self.use_gender_token:
                file.write(f'{indent} cc_combinedusesgender\n')
            if self.suppress_caption_attenuation:
                file.write(f'{indent} cc_noattenuate\n')

        file.write(f'{indent} }}\n')


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
    """A channel defines a set of events that an actor performs."""
    name: str
    active: bool = True
    events: List[Event] = attrs.Factory(list)

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Self:
        """Parse the BVCD form of this data."""
        [name_ind, event_count] = binformat.struct_read('<hB', file)
        name = string_pool[name_ind]
        events = [
            Event.parse_binary(file, string_pool)
            for _ in range(event_count)
        ]
        active = file.read(1) != b'\x00'
        return cls(name, active, events)

    @classmethod
    def parse_text(cls, tokenizer: BaseTokenizer) -> Self:
        """Parse text data. The 'channnel' string should have already been parsed."""
        name = tokenizer.expect(Token.STRING)
        channel = cls(name)
        tokenizer.expect(Token.BRACE_OPEN)
        for tok, tok_str in tokenizer:
            if tok is Token.NEWLINE:
                continue
            if tok is Token.BRACE_CLOSE:
                return channel
            if tok is not Token.STRING:
                raise tokenizer.error(tok, tok_str)
            folded = tok_str.casefold()
            if folded == "event":
                channel.events.append(Event.parse_text(tokenizer))
            elif folded == "active":
                channel.active = conv_bool(tokenizer.expect(Token.STRING))
            else:
                raise tokenizer.error('Unknown channel keyvalue "{}"!', tok_str)
        raise tokenizer.error(Token.EOF)

    def export_binary(self, file: IO[bytes], add_to_pool: Callable[[str], int]) -> None:
        """Write this to a binary BVCD block."""
        file.write(struct.pack('<hB', add_to_pool(self.name), len(self.events)))
        for channel in self.events:
            channel.export_binary(file, add_to_pool)
        file.write(b'\x01' if self.active else b'\x00')

    def export_text(self, file: IO[str], indent: str) -> None:
        """Write this to a text VCD file."""
        file.write(f'{indent}channel "{escape_text(self.name)}"\n')
        file.write(f'{indent} {{\n')
        sub_indent = indent + ' '
        for event in self.events:
            event.export_text(file, sub_indent)
        if not self.active:
            file.write(f'{indent} active "0"\n')
        file.write(f'{indent} }}\n')


@attrs.define(eq=False)
class Actor:
    """An actor in a choreo scene."""
    name: str
    active: bool = True
    channels: List[Channel] = attrs.Factory(list)
    faceposer_model: str = ''

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Self:
        """Parse the BVCD form of this data."""
        [name_ind, channel_count] = binformat.struct_read('<hB', file)
        name = string_pool[name_ind]
        channels = [
            Channel.parse_binary(file, string_pool)
            for _ in range(channel_count)
        ]
        active = file.read(1) != b'\x00'
        return cls(name, active, channels)

    @classmethod
    def parse_text(cls, tokenizer: BaseTokenizer) -> Self:
        """Parse text data. The 'actor' string should have already been parsed."""
        name = tokenizer.expect(Token.STRING)
        actor = cls(name)
        tokenizer.expect(Token.BRACE_OPEN)
        for tok, tok_str in tokenizer:
            if tok is Token.NEWLINE:
                continue
            if tok is Token.BRACE_CLOSE:
                return actor
            if tok is not Token.STRING:
                raise tokenizer.error(tok, tok_str)
            folded = tok_str.casefold()
            if folded == "channel":
                actor.channels.append(Channel.parse_text(tokenizer))
            elif folded == "faceposermodel":
                actor.faceposer_model = tokenizer.expect(Token.STRING)
            elif folded == "active":
                actor.active = conv_bool(tokenizer.expect(Token.STRING))
            else:
                raise tokenizer.error('Unknown actor keyvalue "{}"!', tok_str)
        raise tokenizer.error(Token.EOF)

    def export_binary(self, file: IO[bytes], add_to_pool: Callable[[str], int]) -> None:
        """Write this to a binary BVCD block."""
        file.write(struct.pack('<hB', add_to_pool(self.name), len(self.channels)))
        for channel in self.channels:
            channel.export_binary(file, add_to_pool)
        file.write(b'\x01' if self.active else b'\x00')

    def export_text(self, file: IO[str], indent: str) -> None:
        """Write this to a text VCD file."""
        file.write(f'{indent}actor "{escape_text(self.name)}"\n')
        file.write(f'{indent} {{\n')
        sub_indent = indent + ' '
        for channel in self.channels:
            channel.export_text(file, sub_indent)
        if self.faceposer_model:
            file.write(f'{indent} faceposermodel "{escape_text(self.faceposer_model)}"\n')
        if not self.active:
            file.write(f'{indent} active "0"\n')
        file.write(f'{indent} }}\n')


@attrs.define(eq=False, kw_only=True)
class Scene:
    """A choreo scene."""
    events: List[Event] = attrs.Factory(list)
    actors: List[Actor] = attrs.Factory(list)
    ramp: Curve = attrs.Factory(lambda: Curve([]))
    ignore_phonemes: bool = False
    # CRC for the original VCD that this scene was parsed from.
    text_crc: int = 0

    # VCD only?
    # channels: List[Channel] = attrs.Factory(list)
    # map_name: str = ''
    # fps: int = 0
    # time_zoom_lookup: Dict[int, int] = attrs.Factory(dict)
    # is_background: bool = False
    # is_sub_scene: bool = False
    # use_frame_snap: bool = False

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Self:
        """Parse the BVCD form of this data."""
        if file.read(4) != b'bvcd':
            raise ValueError('File is not a binary VCD scene!')
        version = file.read(1)[0]
        if version != BINARY_VERSION:
            raise ValueError(f'Unknown version "{version}"!')
        [text_crc, event_count] = binformat.struct_read('<IB', file)

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
            text_crc=text_crc,
            ignore_phonemes=ignore_phonemes,
        )

    def export_binary(self, add_to_pool: Callable[[str], int]) -> bytes:
        """Write out BVCD data for this scene."""
        file = BytesIO()
        file.write(struct.pack(
            '<4sbIB', b'bvcd', BINARY_VERSION, self.text_crc, len(self.events),
        ))
        for event in self.events:
            event.export_binary(file, add_to_pool)
        file.write(struct.pack('B', len(self.actors)))
        for actor in self.actors:
            actor.export_binary(file, add_to_pool)
        self.ramp.export_binary(file)
        file.write(b'\x01' if self.ignore_phonemes else b'\x00')
        return file.getvalue()

    def export_text(self, file: IO[str]) -> None:
        """Write this to a text VCD file."""
        file.write('// Choreo version 1\n')
        for event in self.events:
            event.export_text(file, '')
        for actor in self.actors:
            actor.export_text(file, '')
        self.ramp.export_text(file, '', 'scene_ramp')


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

    string_pool = binformat.read_offset_array(file, string_count, 'latin1')
    scenes: ScenesImage = {}

    file.seek(scene_off)
    scene_data: List[Tuple[CRC, int, int, int]] = [
        binformat.struct_read('<Iiii', file)
        for _ in range(scene_count)
    ]

    for (
        crc,
        data_off, data_size,
        summary_off,
    ) in scene_data:
        file.seek(summary_off)
        if version == 3:
            [duration, last_speak, sound_count] = binformat.struct_read('<Iii', file)
        else:
            [duration, sound_count] = binformat.struct_read('<Ii', file)
            last_speak = duration  # Assume it's the whole choreo scene.
        sounds = [
            string_pool[i]
            for i in binformat.struct_read('<{}i'.format(sound_count), file)
        ]
        file.seek(data_off)
        data = binformat.decompress_lzma(file.read(data_size))
        scenes[crc] = Entry(
            '',
            crc,
            duration, last_speak,
            sounds,
            (data, string_pool),
        )
    return scenes


# noinspection PyProtectedMember
def save_scenes_image(
    file: IO[bytes],
    scenes: Union[ScenesImage, Iterable[Entry]],
    *,
    version: Literal[2, 3] = 3,
    encoding: str = 'latin1',
) -> None:
    """Write a new ``scenes.image`` file.
    
    Binary scenes use a common string pool, meaning that all unparsed scenes must share their pool
    to be copied over directly.
    """
    if isinstance(scenes, dict):
        scene_list = list(scenes.values())
    else:
        scene_list = list(scenes)

    # First, loop through and see if we do have a string pool to reuse.
    pool: Optional[List[str]] = None
    two_pools = False
    for entry in scene_list:
        if isinstance(entry._data, Scene):
            continue
        data, entry_pool = entry._data
        if pool is None:
            pool = entry_pool
        elif pool is not entry_pool:
            two_pools = True
            pool = None
            break
    if pool is None:
        pool = []

    add_to_pool = binformat.find_or_insert(pool)
    deferred = binformat.DeferredWrites(file)

    # Now, go through every scene, writing their data so our pool is filled.
    entry_to_data: dict[Entry, bytes] = {}
    for entry in scene_list:
        for sound in entry.sounds:
            add_to_pool(sound)
        if not two_pools and isinstance(entry._data, tuple):
            data, entry_pool = entry._data
            entry_to_data[entry] = data
            assert entry_pool is pool
        else:
            # Parse if required, then export.
            entry_to_data[entry] = entry.data.export_binary(add_to_pool)
    # The entry CRCs must be sorted, since the game uses a binary search.
    scene_list.sort(key=lambda entry: entry.checksum)

    # Finally we can start writing to the file.
    file.write(struct.pack('<4siii', b'VSIF', version, len(scene_list), len(pool)))
    deferred.defer('scene_offset', '<i', write=True)
    # Defer the block of offsets, write the strings, then come back.
    pool_offset_size = len(pool) * binformat.SIZE_INT
    deferred.defer('pool_offsets', f'<{len(pool)}s', write=True)
    offsets = []
    for string in pool:
        offsets.append(file.tell())
        file.write(string.encode(encoding) + b'\x00')
    deferred.set_data('pool_offsets', binformat.write_array('<i', offsets))

    deferred.set_data('scene_offset', file.tell())
    for entry in scene_list:
        file.write(struct.pack('<I', entry.checksum))
        deferred.defer(('data', entry.checksum), '<ii', write=True)
        deferred.defer(('summary', entry.checksum), '<i', write=True)
    # Now write the summaries.
    for entry in scene_list:
        deferred.set_data(('summary', entry.checksum), file.tell())
        if version == 3:
            file.write(struct.pack('<Iii', entry.duration_ms, entry.last_speak_ms, len(entry.sounds)))
        else:
            file.write(struct.pack('<Ii', entry.duration_ms, len(entry.sounds)))
        for sound in entry.sounds:
            file.write(struct.pack('<i', add_to_pool(sound)))
    # Finally, write each data.
    for entry in scene_list:
        data = entry_to_data[entry]
        compressed = binformat.compress_lzma(data)
        if len(compressed) < len(data):
            data = compressed
        deferred.set_data(('data', entry.checksum), file.tell(), len(data))
        file.write(data)

    deferred.write()
    assert len(offsets) == len(pool), 'Pool changed size after being written!'
