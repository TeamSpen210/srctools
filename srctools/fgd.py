"""Parse FGD files, used to describe Hammer entities."""
from enum import Enum
import os.path

from typing import List, Tuple, Dict, Iterator, Iterable, Callable

# Reuse this error for FGD files too.
from srctools import KeyValError


class ValueTypes(Enum):
    """Types which can be applied to a KeyValue."""
    # Special cases:
    VOID = 'void'  # Nothing
    CHOICES = 'choices'  # Special - preset value list as string
    SPAWNFLAGS = 'flags'  # Binary flag values.

    # Simple values
    STRING = 'string'
    BOOLEAN = 'boolean'
    INT = 'int'
    FLOAT = 'float'
    VEC = 'vector'  # Offset or the like
    ANGLES = 'angles'  # Rotation

    # String targetname values (need fixups)
    TARG_DEST = 'target_destination' # A targetname of another ent.
    TARG_DEST_CLASS = 'target_destination_or_class' # Above + classnames.
    TARG_SOURCE = 'target_source'  # The 'targetname' keyvalue.
    TARG_NPC_CLASS = 'npcclass'  # targetnames filtered to NPC ents
    TARG_POINT_CLASS = 'pointentityclass' # targetnames filtered to point enitites.
    TARG_FILTER_NAME = 'filterclass'  # targetnames of filters.
    TARG_NODE = 'node_dest'  # Node entities

    # Strings, don't need fixups
    STR_SCENE = 'scene'  # VCD files
    STR_SOUND = 'sound'  # WAV & SoundScript
    STR_SPRITE = 'sprite'  # Sprite materials
    STR_MATERIAL = 'material'  # Materials
    STR_MODEL = 'studio'  # Model

    # More complex
    VEC_LINE = 'vecline'  # Absolute vector, with line drawn from origin to point
    COLOR_1 = 'color1'  # RGB 0-1 + extra
    COLOR_255 = 'color255'  # RGB 0-255 + extra
    SIDE_LIST = 'sidelist'  # Space-seperated list of sides.


class EntityTypes(Enum):
    BASE = 'baseclass'  # Not an entity, others inherit from this.
    POINT = 'pointclass'  # Point entity
    BRUSH = 'solidclass'  # Brush entity. Can't have 'model'
    ROPES = 'keyframeclass'  # Used for move_rope etc
    TRACK = 'moveclass'  # Used for path_track etc
    FILTER = 'filterclass'  # Used for filters


class EntityDef:
    """A definition for an entity."""
    def __init__(self, type: EntityTypes):
        self.type = type

    @classmethod
    def parse(cls, fgd: 'FGD', file_iter, first_line: str, filename, start_num=0):
        """Parse an entity definition."""


class FGD:
    """A FGD set for a game. May be composed of several files."""
    def __init__(self, directory):
        """Create a FGD."""
        # List of names we have already parsed.
        # We don't parse them again, to prevent infinite loops.
        self._parse_list = []
        # Entity definitions
        self.entities = {}  # type: Dict[str, EntityDef]

        self.root = directory

    @classmethod
    def parse(
        cls,
        filename: str,
        read_func=open,
    ) -> 'FGD':
        """Parse an FGD file.

        Parameters:
        * filename: The name of the file to read. This should be suitable to
            pass to read_func.
        * read_func: Allows reading from locations other than the filesystem.
            Read_func should be a callable which returns a file-like object
            when called with a file path, or raises FileNotFoundError.
        """
        dirname, file = os.path.split(filename)
        fgd = cls(dirname)
        fgd._parse_file(file, read_func)
        return fgd

    def _parse_file(self, file: str, read_func: Callable[[str], Iterable[str]]):
        """Parse one file (recursively if needed)."""
        file = file.replace('\\', '/')
        if not file.endswith('.fgd'):
            file += '.fgd'
        full_path = os.path.join(self.root, file)
        print('Reading "{}"'.format(full_path))

        if file in self._parse_list:
            return

        self._parse_list.append(file)

        with read_func(full_path) as f:
            # Keep the iterator, so other functions can consume it too.
            file_iter = iter(f)
            for line_num, indented_line in enumerate(file_iter):
                line = indented_line.lstrip()
                if not line or line[:2] == '//':
                    continue

                if line.startswith('@include'):
                    if '"' in line:
                        include_file = line.split('"')[1]
                        if line.rstrip()[-1] != '"':
                            raise KeyValError(
                                'Include missing end quote!',
                                full_path,
                                line_num,
                            )
                    else:
                        include_file = line[8:].strip()
                    try:
                        self._parse_file(include_file, read_func)
                    except FileNotFoundError:
                        raise KeyValError(
                            'Cannot include "{}"!'.format(include_file),
                            full_path,
                            line_num,
                        )
                # Entity definition...
                elif line[0] == '@':
                    EntityDef.parse(self, file_iter, line, full_path, line_num)
                else:
                    raise KeyValError(
                        'Unexpected line "{}"'.format(line),
                        full_path,
                        line_num,
                    )

    def __getitem__(self, classname) -> EntityDef:
        try:
            return self.entities[classname.casefold()]
        except KeyError:
            raise KeyError('No class "{}"!'.format(classname)) from None

    def __iter__(self) -> Iterator[EntityDef]:
        return iter(self.entities.values())