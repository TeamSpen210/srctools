"""Parses VCD choreo scenes, as well as data in scenes.image."""
from __future__ import annotations

from io import BytesIO
from typing import List, IO, NewType, Dict, Tuple
from typing_extensions import TypeAlias

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


@attrs.define(eq=False, kw_only=True)
class Event:
    ...


@attrs.define(eq=False, kw_only=True)
class Actor:
    ...


@attrs.define(eq=False, kw_only=True)
class Channel:
    ...


@attrs.define(eq=False, kw_only=True)
class Scene:
    """A choreo scene."""
    events: List[Event]
    actors: List[Actor]
    channels: List[Channel]
    map_name: str
    fps: int
    ramp: object
    time_zoom_lookup: Dict[int, int]
    is_background: bool
    ignore_phonemes: bool
    is_sub_scene: bool
    use_frame_snap: bool

    @classmethod
    def parse_binary(cls, file: IO[bytes], string_pool: List[str]) -> Scene:
        """Parse from binary ``scenes.image`` data."""
        if file.read(4) != b'bvcd':
            raise ValueError('File is not a binary VCD scene!')
        version = file.read(1)[0]
        if version != 4:
            raise ValueError(f'Unknown version "{version}"!')
        [
            crc,
            event_count,
            actor_count,
        ] = binformat.struct_read('<IBB', file)



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
