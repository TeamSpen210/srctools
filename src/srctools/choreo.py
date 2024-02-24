"""Parses VCD choreo scenes, as well as data in scenes.image."""
from typing import List, IO, NewType, Dict, Tuple
import attrs


from srctools import binformat


CRC = NewType('CRC', int)


def checksum_filename(filename: str) -> CRC:
    """Normalise the filename, then checksum it."""
    filename = filename.lower().replace('/', '\\')
    if not filename.startswith('scenes\\'):
        filename = 'scenes\\' + filename
    return CRC(binformat.checksum(filename.encode('ascii')))


@attrs.define
class Choreo:
    """A choreographed scene."""
    filename: str  # Filename if available.
    checksum: CRC  # CRC hash.
    duration_ms: int  # Duration in milliseconds.
    last_speak_ms: int  # Time at which the last voice line ends.
    sounds: List[str]  # List of sounds it uses.
    _data: bytes

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


def parse_scenes_image(file: IO[bytes]) -> Dict[CRC, Choreo]:
    """Parse the scenes.image file, extracting all the choreo data."""
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

    scenes: Dict[CRC, Choreo] = {}

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
            last_speak = duration - 1  # Assume it's the whole choreo.
        sounds = [
            string_pool[i]
            for i in binformat.struct_read('<{}i'.format(sound_count), file)
        ]
        file.seek(data_off)
        data = file.read(data_size)
        if data.startswith(b'LZMA'):
            data = binformat.decompress_lzma(data)
        scenes[crc] = Choreo(
            '',
            crc,
            duration, last_speak,
            sounds,
            data,
        )
    return scenes
