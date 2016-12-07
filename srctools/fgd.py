"""Parse FGD files, used to describe Hammer entities."""
from enum import Enum
import os.path
import re

from typing import List, Tuple, Dict, Iterator, Iterable, Callable

from srctools import KeyValError, FileParseProgress, Vec, clean_line

__all__ = [
    'ValueTypes', 'EntityTypes'
    'KeyValError', 'FGD', 'EntityDef',
]

# "text" +
_RE_DOC_LINE = re.compile(r'\s*"([^"]*)"\s*(\+)?\s*')

_RE_KEYVAL_LINE = re.compile(
    r''' (input | output)? \s* # Input or output name
    (\w+)\s*\(\s*(\w+)\s*\) # Name, (type)
    \s* (report | readonly)?  # Flags for the text
    (?: \s* : \s* \"([^"]*)\"\s* # Display name
        (\+)? \s* # IO only - plus for continued description
        (?::([^:]+)  # Default
            (?::([^:]+)  # Docs
            )?
        )?
    )? # Optional for spawnflags..
    \s* (=)? # Has equal sign?
    ''',
    re.VERBOSE
)


class ValueTypes(Enum):
    """Types which can be applied to a KeyValue."""
    # Special cases:
    VOID = 'void'  # Nothing
    CHOICES = 'choices'  # Special - preset value list as string
    SPAWNFLAGS = 'flags'  # Binary flag values.

    # Simple values
    STRING = 'string'
    BOOL = 'boolean'
    INT = 'integer'
    FLOAT = 'float'
    VEC = 'vector'  # Offset or the like
    ANGLES = 'angle'  # Rotation

    # String targetname values (need fixups)
    TARG_DEST = 'target_destination'  # A targetname of another ent.
    TARG_DEST_CLASS = 'target_name_or_class'  # Above + classnames.
    TARG_SOURCE = 'target_source'  # The 'targetname' keyvalue.
    TARG_NPC_CLASS = 'npcclass'  # targetnames filtered to NPC ents
    TARG_POINT_CLASS = 'pointentityclass'  # targetnames filtered to point entities.
    TARG_FILTER_NAME = 'filterclass'  # targetnames of filters.
    TARG_NODE_DEST = 'node_dest'  # name of a node
    TARG_NODE_SOURCE = 'node_id'  # name of us

    # Strings, don't need fixups
    STR_SCENE = 'scene'  # VCD files
    STR_SOUND = 'sound'  # WAV & SoundScript
    STR_PARTICLE = 'particlesystem'  # Particles
    STR_SPRITE = 'sprite'  # Sprite materials
    STR_DECAL = 'decal'  # Sprite materials
    STR_MATERIAL = 'material'  # Materials
    STR_MODEL = 'studio'  # Model
    STR_VSCIPT = 'scriptlist'  # List of vscripts

    # More complex
    ANGLE_NEG_PITCH = 'angle_negative_pitch'  # Inverse pitch of 'angles'
    VEC_LINE = 'vecline'  # Absolute vector, with line drawn from origin to point
    VEC_ORIGIN = 'origin'  # Used for 'origin' keyvalue
    VEC_AXIS = 'axis'
    COLOR_1 = 'color1'  # RGB 0-1 + extra
    COLOR_255 = 'color255'  # RGB 0-255 + extra
    SIDE_LIST = 'sidelist'  # Space-seperated list of sides.

    # Instances
    INST_FILE = 'instance_file'  # File of func_instance
    INST_VAR_DEF = 'instance_parm'  # $fixup definition
    INST_VAR_REP = 'instance_variable'  # $fixup usage

    @property
    def has_list(self):
        """Is this a flag or choices value, and needs a [] list?"""
        return self.value in ('choices', 'flags')

VALUE_TYPE_LOOKUP = {
    typ.value: typ
    for typ in ValueTypes
}
# These have two names pointing to the same type...
VALUE_TYPE_LOOKUP['bool'] = ValueTypes.BOOL
VALUE_TYPE_LOOKUP['int'] = ValueTypes.INT


class EntityTypes(Enum):
    BASE = 'baseclass'  # Not an entity, others inherit from this.
    POINT = 'pointclass'  # Point entity
    BRUSH = 'solidclass'  # Brush entity. Can't have 'model'
    ROPES = 'keyframeclass'  # Used for move_rope etc
    TRACK = 'moveclass'  # Used for path_track etc
    FILTER = 'filterclass'  # Used for filters
    NPC = 'npcclass'  # An NPC


def read_multiline_desc(file: FileParseProgress, first_line) -> str:
    """Read a docstring from the file.

    These are a quoted string, which can be joined across lines
    via a '+':
    "line 1" +
    "line 2" +
    "line 3"
    This returns a list of lines, and whether the last is followed by an
    equals sign (used by 'choice' keyvalues).
    """
    if first_line is None:
        return ''

    match = _RE_DOC_LINE.match(first_line)
    if match:
        first_line, has_multi = match.groups()
        doc_lines = [first_line]
    elif not first_line:
        has_multi = True
        doc_lines = []
    else:
        return first_line

    if has_multi:
        for line in file:
            match = _RE_DOC_LINE.match(line)
            if match:
                line, has_multi = match.groups()
                doc_lines.append(line)
            else:
                raise file.error(
                    "Documentation line expected, but none found!",
                )
            if not has_multi:
                break

    return '\n'.join(doc_lines)


def read_value_list(file: FileParseProgress):
    """Read the [...] list used for spawnflags or choices keyvalues."""
    values = []
    file.expect('[')
    for line in file:
        line = clean_line(line)
        if not line:
            continue
        if line == ']':
            return values

        if ':' in line:
            key, desc = line.split(':', 1)
            desc = read_multiline_desc(file, desc)
            values.append((key.strip(), desc))
        else:
            raise file.error('No ":" in values list!')
    else:
        raise file.error('EOF when reading value list!')

class KeyValues:
    """Represents a generic keyvalue type."""
    def __init__(self, name, val_type, default, doc, val_list, is_readonly):
        self.name = name
        self.type = val_type
        self.default = default
        self.desc = doc
        self.val_list = val_list
        self.readonly = is_readonly

class IODef:
    """Represents an input or output for an entity."""
    def __init__(self, name, val_type: ValueTypes, description: str):
        self.name = name
        self.type = val_type
        self.desc = description


class EntityDef:
    """A definition for an entity."""
    def __init__(self, type: EntityTypes):
        self.type = type
        self.classname = ''
        self.keyvalues = {}
        self.inputs = {}
        self.outputs = {}
        # Base type names - base()
        self.bases = []
        # line(), studio(), etc in the header
        # this is a func, args tuple.
        self.helpers = []
        self.desc = []

    @classmethod
    def parse(
        cls,
        fgd: 'FGD',
        file: FileParseProgress,
        first_line: str,
    ):
        """Parse an entity definition."""
        try:
            ent_type, first_line = first_line.split(None, 1)
        except ValueError:
            # The entity type might be on its own line..
            ent_type = first_line
            first_line = next(file)
        try:
            ent_type = EntityTypes(ent_type[1:].casefold())
        except ValueError:
            raise file.error(
                'Invalid Entity type "{}"!',
                ent_type[1:],
            )

        start_line_num = file.line_num

        entity = cls(ent_type)

        helpers = []

        ent_name = None
        file.repeat(first_line)
        for line in file:
            if '=' in line:
                line, ent_name = line.split('=', 1)

            helpers.append(line)

            if ent_name is not None:
                if ':' in ent_name:
                    entity.classname, desc = ent_name.split(':', 1)
                    entity.desc = read_multiline_desc(file, desc.strip() or '')
                else:
                    entity.classname = ent_name
                break
        else:
            raise KeyValError(
                'Entity header never ended!',
                file.filename,
                start_line_num,
            )

        file.expect('[')

        # Now parse keyvalues, and input/outputs
        for line in file:
            # Most are only one line, plus docstring - other than flags
            # or choices.
            line = clean_line(line)
            if not line:
                continue

            if line == ']':
                break

            match = _RE_KEYVAL_LINE.match(line)
            if match is None:
                raise file.error('Unrecognised line! ({!r})', line)
            (
                io_type,
                name,
                val_typ,
                is_readonly,
                disp_name,
                io_continuation,
                default,
                desc,
                has_equal,
            ) = match.groups()

            try:
                val_typ = VALUE_TYPE_LOOKUP[val_typ.casefold()]
            except KeyError:
                raise file.error('Unknown keyvalue type "{}"!', val_typ)

            if io_type:
                if desc or default:
                    raise file.error('Too many values for input or output!')
                if is_readonly:
                    raise file.error('Inputs/outputs cannot be readonly!')
                if val_typ.has_list:
                    raise file.error(
                        '"{}" value type is not valid for an input or output!',
                        val_typ.value,
                    )
                if has_equal:
                    raise file.error('Unexpected "=" in input or output!')

                desc = read_multiline_desc(
                    file,
                    '"{}"{}'.format(disp_name, io_continuation or ''),
                )
                io_type = io_type.casefold()
                if io_type == 'input':
                    entity.inputs[name] = IODef(name, val_typ, desc)
                elif io_type == 'output':
                    entity.inputs[name] = IODef(name, val_typ, desc)
                else:
                    raise file.error('"{}" must be input, or output!', io_type)
            else:
                desc = read_multiline_desc(file, desc)
                if val_typ.has_list:
                    # It's a flags or choices value - read in the list following
                    val_list = read_value_list(file)
                else:
                    val_list = None

                entity.keyvalues[name] = KeyValues(
                    val_typ,
                    disp_name,
                    default,
                    desc,
                    val_list,
                    is_readonly == 'readonly',
                )

class FGD:
    """A FGD set for a game. May be composed of several files."""
    def __init__(self, directory):
        """Create a FGD."""
        # List of names we have already parsed.
        # We don't parse them again, to prevent infinite loops.
        self._parse_list = []
        # Entity definitions
        self.entities = {}  # type: Dict[str, EntityDef]
        # maximum bounding box of map
        self.map_size_min = 0
        self.map_size_max = 0

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

    def _parse_file(self, filename: str, read_func: Callable[[str], Iterable[str]]):
        """Parse one file (recursively if needed)."""
        filename = filename.replace('\\', '/')
        if not filename.endswith('.fgd'):
            filename += '.fgd'
        full_path = os.path.join(self.root, filename)
        print('Reading "{}"'.format(full_path))

        if filename in self._parse_list:
            return

        self._parse_list.append(filename)

        with read_func(full_path) as f:
            # Keep the iterator, so other functions can consume it too.
            file = FileParseProgress(f, full_path)
            for indented_line in file:
                line = clean_line(indented_line)
                if not line:
                    continue
                folded_line = line.casefold()

                if folded_line.startswith('@include'):
                    if '"' in line:
                        include_file = line.split('"')[1]
                        if line.rstrip()[-1] != '"':
                            file.error('@include missing end quote!')
                    else:
                        include_file = line[8:].strip()
                    try:
                        self._parse_file(include_file, read_func)
                    except FileNotFoundError:
                        raise file.error(
                            'Cannot include "{}"!',
                            include_file,
                        )
                elif folded_line.startswith('@mapsize'):
                    # Max/min map size definition
                    try:
                        min_size, max_size = folded_line[9:].split(',')
                        self.map_size_min = int(min_size)
                        self.map_size_min = int(max_size.strip('\n )'))
                    except ValueError:
                        raise file.error(
                            'Invalid @MapSize! ("")',
                            line,
                        )
                # Entity definition...
                elif line[0] == '@':
                    EntityDef.parse(self, file, line)
                else:
                    raise file.error(
                        'Unexpected line "{}"',
                        line,
                    )

    def __getitem__(self, classname) -> EntityDef:
        try:
            return self.entities[classname.casefold()]
        except KeyError:
            raise KeyError('No class "{}"!'.format(classname)) from None

    def __iter__(self) -> Iterator[EntityDef]:
        return iter(self.entities.values())

f = FGD.parse(r'F:\Git\HammerAddons\bin\portal2.fgd')