"""Parse FGD files, used to describe Hammer entities."""
from copy import deepcopy
from collections import defaultdict
from enum import Enum
from pathlib import PurePosixPath
from struct import Struct
from functools import partial
import itertools
import io
import math
import operator

from typing import (
    Generic, Optional, TYPE_CHECKING, Union, overload, cast, ClassVar, Any,
    TypeVar, Callable, Type,
    Dict, Tuple, List, Set, FrozenSet,
    Mapping, Iterator, Iterable, Collection,
    TextIO, Container, IO,
)

import attr

import srctools
from srctools.filesys import FileSystem, File
from srctools.tokenizer import (
    BaseTokenizer, Tokenizer, Token,
    TokenSyntaxError, escape_text,
)
from srctools.binformat import struct_read

try:
    from importlib.resources import open_binary
except ImportError:
    # Backport module for before Python 3.7
    from importlib_resources import open_binary  # type: ignore

__all__ = [
    'ValueTypes', 'EntityTypes', 'HelperTypes',
    'FGD', 'EntityDef', 'KeyValues', 'IODef', 'Helper', 'UnknownHelper', 'AutoVisgroup',
    'match_tags', 'validate_tags',

    # From srctools._fgd_helpers
    'HelperBBox', 'HelperBoundingBox', 'HelperBreakableSurf',
    'HelperBrushSides', 'HelperCylinder', 'HelperDecal',
    'HelperEnvSprite', 'HelperFrustum', 'HelperHalfGridSnap',
    'HelperInherit', 'HelperInstance', 'HelperLight', 'HelperLightSpot',
    'HelperLine', 'HelperModel', 'HelperModelLight', 'HelperModelProp',
    'HelperOrientedBBox', 'HelperOrigin', 'HelperOverlay',
    'HelperOverlayTransition', 'HelperRenderColor', 'HelperRope',
    'HelperSize', 'HelperSphere', 'HelperSprite',
    'HelperSweptPlayerHull', 'HelperTrack', 'HelperTypes',
    'HelperVecLine', 'HelperWorldText',

    'HelperExtAppliesTo', 'HelperExtAutoVisgroups', 'HelperExtOrderBy',
]

_fmt_8bit = Struct('>B')
_fmt_16bit = Struct('>H')
_fmt_32bit = Struct('>I')
_fmt_double = Struct('>d')
_fmt_header = Struct('>BddI')
_fmt_ent_header = Struct('<BBBBBB')


# Version number for the format.
BIN_FORMAT_VERSION = 5
# Cached result of FGD.engine_dbase().
_ENGINE_FGD: Optional['FGD'] = None
T = TypeVar('T')


class FGDParseError(TokenSyntaxError):
    """Raised if the FGD contains invalid syntax."""


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

    # A targetname of another ent. For outputs, this is an entity handle.
    TARG_DEST = 'target_destination'
    ENT_HANDLE = EHANDLE = TARG_DEST

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
    STR_VSCRIPT = 'scriptlist'  # List of vscripts
    STR_VSCRIPT_SINGLE = 'script'  # Single VScript path.

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

    # Format extensions.
    EXT_STR_TEXTURE = 'texture'  # A VTF, mainly for env_projectedtexture.
    EXT_VEC_DIRECTION = 'vec_dir'  # A vector which should be rotated, but not translated.
    EXT_VEC_LOCAL = 'vec_local'  # Vector, but do not rotate in instances.
    EXT_ANGLE_PITCH = 'angle_pitch'  # Overrides angles[2], but isn't inverted
    EXT_ANGLES_LOCAL = 'angle_local'  # Angles value, but do not rotate in instances.

    @property
    def has_list(self) -> bool:
        """Is this a flag or choices value, and needs a [] list?"""
        return self.value in ('choices', 'flags')

    @property
    def valid_for_io(self) -> bool:
        """Is this type valid for I/O definitions?"""
        return self.value in {
            'void',
            'integer', 'boolean', 'string', 'float', 'script',
            'vector', 'target_destination', 'color255'
        }

    @property
    def extension(self) -> bool:
        """Is this an extension to the format?"""
        return self.name.startswith('EXT_')

    @property
    def is_ent_name(self) -> bool:
        """Several types are simply a targetname."""
        return self.value in {
            'target_source', 'target_destination', 'npcclass',
            'pointentityclass', 'filterclass',
        }


VALUE_TYPE_LOOKUP: Dict[str, ValueTypes] = {
    typ.value: typ
    for typ in ValueTypes
}
# These have two names pointing to the same type...
VALUE_TYPE_LOOKUP['bool'] = ValueTypes.BOOL
VALUE_TYPE_LOOKUP['int'] = ValueTypes.INT

# In I/O definitions, types are constrained.
# So this is the most appropriate type to use instead for all types.
# First, string can reproduce everything if not allowed.
VALUE_TO_IO_DECAY: Dict[ValueTypes, ValueTypes] = {
    typ: typ if typ.valid_for_io else ValueTypes.STRING
    for typ in ValueTypes
}

# Then manually set others.
VALUE_TO_IO_DECAY[ValueTypes.SPAWNFLAGS] = ValueTypes.INT
VALUE_TO_IO_DECAY[ValueTypes.TARG_NODE_SOURCE] = ValueTypes.INT
VALUE_TO_IO_DECAY[ValueTypes.ANGLE_NEG_PITCH] = ValueTypes.FLOAT
VALUE_TO_IO_DECAY[ValueTypes.EXT_ANGLE_PITCH] = ValueTypes.FLOAT

VALUE_TO_IO_DECAY[ValueTypes.VEC_LINE] = ValueTypes.VEC
VALUE_TO_IO_DECAY[ValueTypes.VEC_ORIGIN] = ValueTypes.VEC
VALUE_TO_IO_DECAY[ValueTypes.VEC_AXIS] = ValueTypes.VEC
VALUE_TO_IO_DECAY[ValueTypes.EXT_VEC_DIRECTION] = ValueTypes.VEC
VALUE_TO_IO_DECAY[ValueTypes.EXT_VEC_LOCAL] = ValueTypes.VEC
VALUE_TO_IO_DECAY[ValueTypes.ANGLES] = ValueTypes.VEC
VALUE_TO_IO_DECAY[ValueTypes.EXT_ANGLES_LOCAL] = ValueTypes.VEC
# Only one color type present.
VALUE_TO_IO_DECAY[ValueTypes.COLOR_1] = ValueTypes.COLOR_255


class EntityTypes(Enum):
    """The kind of entity each definition is."""
    BASE = 'baseclass'  # Not an entity, others inherit from this.
    POINT = 'pointclass'  # Point entity
    BRUSH = 'solidclass'  # Brush entity. Can't have 'model'
    ROPES = 'keyframeclass'  # Used for move_rope etc
    TRACK = 'moveclass'  # Used for path_track etc
    FILTER = 'filterclass'  # Used for filters
    NPC = 'npcclass'  # An NPC


class HelperTypes(Enum):
    """Types of functions in the entity header."""
    INHERIT = 'base'

    # Snap to 1/2 of grid.
    # Special - no arguments.
    HALF_GRID_SNAP = 'halfgridsnap'

    # Simple helpers
    CUBE = 'size'  # Sets size of purple cube
    BBOX = 'bbox'  # Sets bounding box of entity
    TINT = 'color'
    SPHERE = 'sphere'
    LINE = 'line'
    FRUSTUM = 'frustum'
    CYLINDER = 'cylinder'
    ORIGIN = 'origin'  # Adds circle at an absolute position.
    VECLINE = 'vecline'  # Draws line to an absolute position.
    BRUSH_SIDES = 'sidelist'  # Highlights brush faces.
    BOUNDING_BOX_HELPER = 'wirebox'  # Displays bounding box from two keyvalues
    # Draws the movement of a player-sized bounding box from A to B.
    SWEPT_HULL = 'sweptplayerhull'
    ORIENTED_BBOX = 'obb'  # Bounding box oriented to angles.

    # Complex helpers using resources
    SPRITE = 'iconsprite'
    MODEL = 'studio'
    MODEL_PROP = 'studioprop'
    MODEL_NEG_PITCH = 'lightprop'  # Uses separate pitch keyvalue

    # Specialty for certain ents
    ENT_SPRITE = 'sprite'
    ENT_INSTANCE = 'instance'
    ENT_DECAL = 'decal'
    ENT_OVERLAY = 'overlay'
    ENT_OVERLAY_WATER = 'overlay_transition'
    ENT_LIGHT = 'light'
    ENT_LIGHT_CONE = 'lightcone'
    ENT_ROPE = 'keyframe'
    ENT_TRACK = 'animator'
    ENT_BREAKABLE_SURF = 'quadbounds'  # Sets the 4 corners on save
    ENT_WORLDTEXT = 'worldtext'  # Renders 3D text in-world.
    ENT_CATAPULT = 'catapult'  # Renders trigger_catpault trajectors prediction

    ENT_LIGHT_CONE_BLACK_MESA = 'lightconenew'  # New helper added in Black Mesa

    # Format extensions.

    # Indicates this entity is only available in the given games.
    EXT_APPLIES_TO = 'appliesto'
    EXT_ORDERBY = 'orderby'  # Reorder keyvalues. Args = names in order.
    # Convenience only used in parsing, adds @AutoVisgroup parents for the
    # current entity. 'Auto' is implied at the start.
    EXT_AUTO_VISGROUP = 'autovis'

    @property
    def extension(self) -> bool:
        """Is this an extension to the format?"""
        return self.name.startswith('EXT_')


# Ordered list of value types, for encoding in the binary
# format. All must be here, new ones should be added at the end.
VALUE_TYPE_ORDER = [
    ValueTypes.VOID,
    ValueTypes.CHOICES,
    ValueTypes.SPAWNFLAGS,

    ValueTypes.STRING,
    ValueTypes.BOOL,
    ValueTypes.INT,
    ValueTypes.FLOAT,
    ValueTypes.VEC,
    ValueTypes.ANGLES,

    ValueTypes.TARG_DEST,
    ValueTypes.TARG_DEST_CLASS,
    ValueTypes.TARG_SOURCE,
    ValueTypes.TARG_NPC_CLASS,
    ValueTypes.TARG_POINT_CLASS,
    ValueTypes.TARG_FILTER_NAME,
    ValueTypes.TARG_NODE_DEST,
    ValueTypes.TARG_NODE_SOURCE,

    # Strings, don't need fixups
    ValueTypes.STR_SCENE,
    ValueTypes.STR_SOUND,
    ValueTypes.STR_PARTICLE,
    ValueTypes.STR_SPRITE,
    ValueTypes.STR_DECAL,
    ValueTypes.STR_MATERIAL,
    ValueTypes.STR_MODEL,
    ValueTypes.STR_VSCRIPT,

    ValueTypes.ANGLE_NEG_PITCH,
    ValueTypes.VEC_LINE,
    ValueTypes.VEC_ORIGIN,
    ValueTypes.VEC_AXIS,
    ValueTypes.COLOR_1,
    ValueTypes.COLOR_255,
    ValueTypes.SIDE_LIST,

    ValueTypes.INST_FILE,
    ValueTypes.INST_VAR_DEF,
    ValueTypes.INST_VAR_REP,

    ValueTypes.STR_VSCRIPT_SINGLE,
    ValueTypes.ENT_HANDLE,
    ValueTypes.EXT_STR_TEXTURE,
    ValueTypes.EXT_ANGLE_PITCH,
    ValueTypes.EXT_ANGLES_LOCAL,
    ValueTypes.EXT_VEC_DIRECTION,
    ValueTypes.EXT_VEC_LOCAL,
]

# Ditto for entity types.
ENTITY_TYPE_ORDER = [
    EntityTypes.BASE,
    EntityTypes.POINT,
    EntityTypes.BRUSH,
    EntityTypes.ROPES,
    EntityTypes.TRACK,
    EntityTypes.FILTER,
    EntityTypes.NPC,
]

assert set(VALUE_TYPE_ORDER) == set(ValueTypes), \
    "Missing values: " + repr(set(ValueTypes) - set(VALUE_TYPE_ORDER))
assert set(ENTITY_TYPE_ORDER) == set(EntityTypes), \
    "Missing values: " + repr(set(EntityTypes) - set(ENTITY_TYPE_ORDER))

# Can only store this many in the bytes.
assert len(VALUE_TYPE_ORDER) < 127, "Too many values."
assert len(ENTITY_TYPE_ORDER) < 255, "Too many entity types."

VALUE_TYPE_INDEX = {val: ind for (ind, val) in enumerate(VALUE_TYPE_ORDER)}
ENTITY_TYPE_INDEX = {ent: ind for (ind, ent) in enumerate(ENTITY_TYPE_ORDER)}


def read_colon_list(tok: BaseTokenizer, had_colon=False) -> Tuple[List[str], Token]:
    """Read strings seperated by colons, up to the end of the line.

    The token found at the end is returned.
    """
    strings = []
    ready_for_string = had_colon  # Did we have a colon before?
    token = Token.EOF
    for token, tok_value in tok:
        if token is Token.STRING:
            if not ready_for_string:
                raise tok.error('Too many strings ({!r})!', tok_value)
            strings.append(tok_value)
            ready_for_string = False
        elif token is Token.COLON:
            if ready_for_string:
                # ': :' means to have an empty string there.
                strings.append('')
            ready_for_string = True
        elif token is Token.PLUS:
            if ready_for_string or not strings:
                raise tok.error('"+" without a string before it!')
            strings[-1] += tok.expect(Token.STRING)
        elif ready_for_string and token is Token.NEWLINE:
            continue  # skip over this in particular..
        else:
            if ready_for_string:
                raise tok.error(token)
            return strings, token
    else:
        raise tok.error(token)


def _write_longstring(file: IO[str], text: str, *, indent: str) -> None:
    """Write potentially long strings to the file, splitting with + if needed.

    The game parser has a max size of 8192 bytes for the text, but can only
    handle 1024 bytes of text in a string. So we need to split after that.
    """
    LIMIT = 1000  # Give a bit of extra room for the quotes, etc.
    sections = []
    remaining = escape_text(text)
    while len(remaining) > LIMIT:
        # First, look for any \ns and split on those. This is a nice stopping
        # point, and also prevents separating the "\" from "n". Then add 2
        # so we leave the \n in the first block.
        split_pos = remaining.rfind('\\n', 0, LIMIT) + 2
        if split_pos > 128:  # Don't do for very small sections.
            sections.append(f'"{remaining[:split_pos]}"')
            remaining = remaining[split_pos:]
            continue
        # Next try splitting at any whitespace, and then leave that in the
        # first block.
        split_pos = remaining.rfind(' ', 0, LIMIT) + 1
        if split_pos == (-1 + 1):
            # Not found, just split exactly at the end.
            split_pos = LIMIT
        sections.append(f'"{remaining[:split_pos]}"')
        remaining = remaining[split_pos:]

    # Lastly add any remaining text that didn't get split off.
    if remaining:
        sections.append(f'"{remaining}"')

    file.write((' +\n' + indent).join(sections))


def read_tags(tok: BaseTokenizer) -> FrozenSet[str]:
    """Parse a set of tags from the file.

    The open bracket was just read.
    """
    tags = []
    prefix = ''
    # Read tags.
    while True:
        token, value = tok()
        if token is Token.STRING:
            tags.append(prefix + value.casefold())
            prefix = ''
        elif token is Token.PLUS:
            if prefix:
                raise tok.error('Duplicate "+" operators!')
            prefix = '+'
        elif token is Token.BRACK_CLOSE:
            break
        elif token is Token.EOF:
            raise tok.error('Unclosed tags!')
        elif token is Token.COMMA:
            continue
        else:
            raise tok.error(token)

    if prefix:
        raise tok.error('Trailing "+" operator!')
    return validate_tags(tags, tok.error)


def validate_tags(
    tags: Collection[str],
    error: Callable[[str], BaseException]=ValueError,
) -> FrozenSet[str]:
    """Check these tags have valid values.

    The error exception is raised if invalid.
    """
    temp_set = {
        t.lstrip('!-+').upper()
        for t in tags
    }
    if len(temp_set) != len(tags):
        raise error('Duplicate tags!')
    if '<any>' in temp_set:
        raise error('<any> cannot be used as a tag!')
    return frozenset({
        t.upper()
        for t in tags
    })


def match_tags(search: Container[str], tags: Iterable[str]):
    """Check if the search constraints satisfy tags.

    The search tags should be uppercased.

    All !tags or -tags cannot be present, all +tags must be present, and
    at lest one normal tag must be present (if they are) to pass.
    """
    if not tags:
        return True

    has_all = '<ALL>' in search
    # None = no normal tags, True = matched one, False = not matched one.
    matched = None
    for tag in tags:
        tag = tag.upper()
        start = tag[0:1]
        if start == '!' or start == '-':
            if tag[1:] in search:
                return False
        elif start == '+':
            if tag[1:] not in search:
                return False
        else:
            if matched is None:
                matched = False
            if has_all or tag in search:
                matched = True

    return matched is not False


class BinStrDict:
    """Manages a "dictionary" for compressing repeated strings in the binary format.

    Each unique string is assigned a 2-byte index into the list.
    """

    def __init__(self) -> None:
        self._dict: Dict[str, int] = {}
        self.cur_index = 0

    def __call__(self, string: str) -> bytes:
        """Get the index for a string.

        If not already present it is assigned one.
        The result is the two bytes that represent the string.
        """
        try:
            index = self._dict[string]
        except KeyError:
            index = self._dict[string] = self.cur_index
            self.cur_index += 1
            # Check it can actually fit.
            if index > (1 << 16):
                raise ValueError("Too many items in dictionary!")

        return _fmt_16bit.pack(index)

    def serialise(self, file: IO[bytes]) -> None:
        """Convert this to a stream of bytes."""
        inv_list = [''] * len(self._dict)
        for txt, ind in self._dict.items():
            inv_list[ind] = txt

        file.write(_fmt_32bit.pack(len(inv_list)))
        for txt in inv_list:
            encoded = txt.encode('utf8')
            file.write(_fmt_16bit.pack(len(encoded)))
            file.write(encoded)

    @staticmethod
    def unserialise(file: IO[bytes]) -> Callable[[], str]:
        """Read the dictionary from a file.

        This returns a function which reads
        a string from a file at the current point.
        """
        [length] = struct_read(_fmt_32bit, file)
        inv_list = [''] * length
        for ind in range(length):
            [str_len] = _fmt_16bit.unpack(file.read(2))
            inv_list[ind] = file.read(str_len).decode('utf8')

        def lookup() -> str:
            """Read the index from the file, and return the string it matches."""
            [index] = _fmt_16bit.unpack(file.read(2))
            return inv_list[index]

        return lookup

    @staticmethod
    def read_tags(file: IO[bytes], from_dict: Callable[[], str]) -> FrozenSet[str]:
        """Pull tags from a BinStrDict."""
        [size] = _fmt_8bit.unpack(file.read(1))
        return frozenset({
            from_dict()
            for _ in range(size)
        })

    @staticmethod
    def write_tags(
        file: IO[bytes],
        dic: 'BinStrDict',
        tags: Collection[str],
    ) -> None:
        """Write tags a file using the dictionary."""
        file.write(_fmt_8bit.pack(len(tags)))
        for tag in tags:
            file.write(dic(tag))


class Helper:
    """Base class for representing helper() commands in the header of an entity.

    These mainly add visual widgets in Hammer's views for manipulating and
    previewing keyvalues.

    This should not be instantiated, only subclasses in _fgd_helpers.
    """
    # The HelperType which this implements.
    TYPE: ClassVar[Optional[HelperTypes]] = None
    IS_EXTENSION: ClassVar[bool] = False  # true for our extensions to the format.

    @classmethod
    def parse(cls: Type['HelperT'], args: List[str]) -> 'HelperT':
        """Parse this helper from the given arguments.

        The default implementation expects no arguments.
        """
        if args:
            raise ValueError('No arguments accepted by {}()!'.format(
                cls.TYPE.name if cls.TYPE is not None else cls.__name__
            ))
        return cls()

    def export(self) -> List[str]:
        """Produce the argument text to recreate this helper type."""
        return []

    def get_resources(self, entity: 'EntityDef') -> Iterable[str]:
        """Return the resources used by this helper."""
        return ()

    def overrides(self) -> Collection[HelperTypes]:
        """Specify which types can be overriden by this.

        If any of these helper types are present before this type, they're
        redundant and can be removed.
        For example size() is ignored if a studio() is present after it.
        """
        return ()

    def __eq__(self, other: object) -> bool:
        """Define equality as all attributes matching, and only matching types."""
        if not isinstance(other, Helper):
            return NotImplemented
        return self.TYPE is other.TYPE and vars(self) == vars(other)

    def __ne__(self, other: object) -> bool:
        """Define equality as all attributes matching, and only matching types."""
        if not isinstance(other, Helper):
            return NotImplemented
        return self.TYPE is not other.TYPE or vars(self) != vars(other)


class UnknownHelper(Helper):
    """Represents an unknown helper."""
    TYPE = None
    def __init__(self, name: str, args: List[str]) -> None:
        """Unknown helpers have a name attribute."""
        self.name = name
        self.args = args

    def export(self) -> List[str]:
        """Produce the argument text to recreate this helper type."""
        return self.args[:]


HelperT = TypeVar('HelperT', bound=Helper)
# Each helper type -> the class implementing them.
# We fill this at the end of the module.
HELPER_IMPL: Dict[HelperTypes, Type[Helper]] = {}


def _init_helper_impl() -> None:
    """Import and register the helper implementations."""
    # noinspection PyProtectedMember
    from srctools import _fgd_helpers as helper_mod
    for helper_type in vars(helper_mod).values():
        if isinstance(helper_type, type) and issubclass(helper_type, Helper):
            if helper_type.TYPE is not None:
                HELPER_IMPL[helper_type.TYPE] = helper_type

    for helper in HelperTypes:
        if helper not in HELPER_IMPL:
            raise ValueError(
                'Missing helper implementation '
                'for {}!'.format(helper)
            )

_init_helper_impl()
del _init_helper_impl
# noinspection PyProtectedMember
from srctools._fgd_helpers import *


@attr.define(order=True, hash=True, eq=True)
class AutoVisgroup:
    """Represents one of the autovisgroup options that can be set.

    Due to how these are coded into Hammer, our representation is rather strange.
    We put all the groups into a single dictionary, and on each specify the name
    of the parent. Note they're case-sensitive, and can include punctuation.
    """
    name: str
    parent: str = attr.ib(hash=False, eq=False, order=False)
    ents: Set[str] = attr.ib(factory=set, hash=False, eq=False, order=False)

    def __repr__(self) -> str:
        return '<AutoVisgroup "{}">'.format(self.name)


@attr.define
class KeyValues:
    """Represents a generic keyvalue type.

    If the type is choices or spawnflags, val_list is required:
    * For choices it's a list of (value, name, tags) tuples.
    * For spawnflags it's a list of (bitflag, name, default, tags) tuples.
    """
    name: str
    type: ValueTypes
    disp_name: str
    default: str = ''
    desc: str = ''
    val_list: Union[
        None,
        List[Tuple[int, str, bool, FrozenSet[str]]],
        List[Tuple[str, str, FrozenSet[str]]],
    ] = None
    readonly: bool = False
    reportable: bool = False

    @property
    def choices_list(self) -> List[Tuple[str, str, FrozenSet[str]]]:
        """Check that the keyvalues are CHOICES type, and then return val_list.

        This isolates the type ambiguity of the attr.
        """
        if self.type is not ValueTypes.CHOICES:
            raise TypeError
        if self.val_list is None:
            lst: List[Tuple[str, str, FrozenSet[str]]] = []
            self.val_list = lst
        return cast(list, self.val_list)

    @property
    def flags_list(self) -> List[Tuple[int, str, bool, FrozenSet[str]]]:
        """Check that the keyvalues are SPAWNFLAGS type, and then return val_list.

        This isolates the type ambiguity of the attr.
        """
        if self.type is not ValueTypes.SPAWNFLAGS:
            raise TypeError
        if self.val_list is None:
            lst: List[Tuple[int, str, bool, FrozenSet[str]]] = []
            self.val_list = lst
        return cast(list, self.val_list)

    def copy(self) -> 'KeyValues':
        """Create a duplicate of this keyvalue."""
        return KeyValues(
            self.name,
            self.type,
            self.desc,
            self.disp_name,
            self.default,
            self.val_list.copy() if self.val_list else None,
            self.readonly,
            self.reportable,
        )

    __copy__ = copy

    def __deepcopy__(self, memodict: dict) -> 'KeyValues':
        return KeyValues(
            self.name,
            self.type,
            self.disp_name,
            self.default,
            self.desc,
            self.val_list.copy() if self.val_list else None,
            self.readonly,
            self.reportable,
        )

    def known_options(self) -> Iterator[str]:
        """Use the default value and value list to determine values this can be set to."""
        if self.type is ValueTypes.CHOICES:
            options = {val_list[0] for val_list in self.choices_list}
            options.add(self.default)
            yield from options
        elif self.type is ValueTypes.SPAWNFLAGS:
            for bitflag, name, default, tags in self.flags_list:
                yield str(bitflag)
        else:
            yield self.default

    def export(self, file: TextIO, tags: Collection[str]=()) -> None:
        """Write this back out to a FGD file."""
        file.write('\t' + self.name)
        if tags:
            file.write('[' + ', '.join(tags) + ']')
        file.write('({}) '.format(self.type.value))

        if self.readonly:
            file.write('readonly ')

        if self.reportable:
            file.write('report ')

        if self.type is not ValueTypes.SPAWNFLAGS:
            # Spawnflags never use names!
            file.write(': "{}"'.format(self.disp_name))

        if self.default:
            default_str = str(self.default)
            # We can write unquoted integers, but nothing else.
            if all(x in '0123456789-' for x in default_str):
                file.write(' : ' + default_str)
            else:
                file.write(' : "{}"'.format(default_str))
            if self.desc:
                file.write(' : ')
        else:
            if self.desc:
                file.write(' : : ')

        if self.desc:
            _write_longstring(file, self.desc, indent='\t')

        if self.type.has_list:
            file.write(' =\n\t\t[\n')
            if self.type is ValueTypes.SPAWNFLAGS:
                # Empty tuple handles a None value.
                for index, name, default, tags in self.flags_list:
                    file.write(f'\t\t{index}: ')
                    # Newlines aren't functional here, just replace.
                    _write_longstring(file, f'[{index}] ' + name.replace('\n', ' '), indent='\t\t')
                    file.write(' : 1' if default else ' : 0')
                    if tags:
                        file.write(' [' + ', '.join(tags) + ']\n')
                    else:
                        file.write('\n')
            elif self.type is ValueTypes.CHOICES:
                for value, name, tags in self.choices_list:
                    # Numbers can be unquoted, everything else cannot.
                    try:
                        float(value)
                    except ValueError:
                        value = '"' + value + '"'

                    file.write(f'\t\t{value}: ')
                    # Newlines aren't functional here, just replace.
                    _write_longstring(file, name.replace('\n', ' '), indent='\t\t')
                    if tags:
                        file.write(' [' + ', '.join(tags) + ']\n')
                    else:
                        file.write('\n')
            else:
                raise AssertionError('No other types possible!')
            file.write('\t\t]\n')

        file.write('\n')

    def serialise(self, file, str_dict: BinStrDict):
        """Write to the binary file."""
        file.write(str_dict(self.name))
        file.write(str_dict(self.disp_name))
        value_type = VALUE_TYPE_INDEX[self.type]
        # Use the high bit to store this inside here as well.
        if self.readonly:
            value_type |= 128
        file.write(_fmt_8bit.pack(value_type))

        # Spawnflags have integer names and defaults,
        # choices has string values and no default.
        if self.type is ValueTypes.SPAWNFLAGS:
            file.write(_fmt_8bit.pack(len(self.flags_list)))
            # spawnflags go up to at least 1<<23.
            for mask, name, default, tags in self.flags_list:
                BinStrDict.write_tags(file, str_dict, tags)
                # We can write 2^n instead of the full number,
                # since they're all powers of two.
                power = int(math.log2(mask))
                assert power < 128, "Spawnflags are too big for packing into a byte!"
                if default:  # Pack the default as the MSB.
                    power |= 128
                file.write(_fmt_8bit.pack(power))
                file.write(str_dict(name))
            return  # Spawnflags doesn't need to write a default.

        file.write(str_dict(self.default or ''))

        if self.type is ValueTypes.CHOICES:
            # Use two bytes, these can be large (soundscapes).
            file.write(_fmt_16bit.pack(len(self.choices_list)))
            for val, name, tags in self.choices_list:
                BinStrDict.write_tags(file, str_dict, tags)
                file.write(str_dict(val))
                file.write(str_dict(name))

    @staticmethod
    def unserialise(
        file: IO[bytes],
        from_dict: Callable[[], str],
    ) -> 'KeyValues':
        """Recover a KeyValue from a binary file."""
        name = from_dict()
        disp_name = from_dict()
        [value_ind] = struct_read(_fmt_8bit, file)
        readonly = value_ind & 128
        value_type = VALUE_TYPE_ORDER[value_ind & 127]

        val_list: Optional[List[tuple]] = None

        if value_type is ValueTypes.SPAWNFLAGS:
            default = ''  # No default for this type.
            [val_count] = struct_read(_fmt_8bit, file)
            val_list = [()] * val_count
            for ind in range(val_count):
                tags = BinStrDict.read_tags(file, from_dict)
                [power] = struct_read(_fmt_8bit, file)
                val_name = from_dict()
                val_list[ind] = (
                    1 << (power & 127),
                    val_name,
                    (power & 128) != 0,
                    tags,
                )
        else:
            default = from_dict()

            if value_type is ValueTypes.CHOICES:
                [val_count] = struct_read(_fmt_16bit, file)
                val_list = [()] * val_count
                for ind in range(val_count):
                    tags = BinStrDict.read_tags(file, from_dict)
                    val_list[ind] = (from_dict(), from_dict(), tags)

        return KeyValues(
            name=name,
            type=value_type,
            disp_name=disp_name,
            default=default,
            val_list=val_list,  # type: ignore
            readonly=readonly,
        )


@attr.define
class IODef:
    """Represents an input or output for an entity."""
    name: str
    type: ValueTypes
    desc: str = ''

    def __str__(self) -> str:
        txt = '{}({!r}, {!r}'.format(
            self.__class__.__name__,
            self.name,
            self.type,
        )
        if self.desc:
            txt += ', ' + repr(self.desc)
        return txt + ')'

    def copy(self) -> 'IODef':
        """Create a duplicate of this IODef."""
        return IODef(self.name, self.type, self.desc)

    __copy__ = copy

    def __deepcopy__(self, memodict: dict) -> 'IODef':
        return IODef(self.name, self.type, self.desc)

    def export(
        self,
        file: TextIO,
        io_type: str,
        tags: Collection[str]=(),
    ) -> None:
        """Write this back out to a FGD file.

        io_type should be "input" or "output".
        """
        file.write('\t{} {}'.format(
            io_type,
            self.name,
        ))

        if tags:
            file.write('[' + ', '.join(tags) + ']')

        # Special case, bool is "boolean" on values, "bool" on IO...
        if self.type is ValueTypes.BOOL:
            file.write('(bool)')
        else:
            file.write('({})'.format(VALUE_TO_IO_DECAY[self.type].value))

        if self.desc:
            file.write(' : ')
            _write_longstring(file, self.desc, indent='\t')
        file.write('\n')

    def serialise(self, file: IO[bytes], dic: BinStrDict) -> None:
        """Write to the binary file."""
        file.write(dic(self.name))
        file.write(_fmt_8bit.pack(VALUE_TYPE_INDEX[self.type]))

    @staticmethod
    def unserialise(
        file: IO[bytes],
        from_dict: Callable[[], str],
    ) -> 'IODef':
        """Recover an IODef from a binary file."""
        name = from_dict()
        value_type = VALUE_TYPE_ORDER[struct_read(_fmt_8bit, file)[0]]
        return IODef(name, value_type)


class _EntityView(Generic[T]):
    """Provides a view over entity keyvalues, inputs, or outputs."""
    __slots__ = ['_ent', '_attr', '_disp_attr']

    # Note, we expect the maps to have casefolded their keys.
    def __init__(self, ent: 'EntityDef', attr_name: str, disp_name: str) -> None:
        self._ent = ent
        self._attr = attr_name
        self._disp_attr = disp_name

    @property
    def __name__(self) -> str:
        return self._disp_attr

    def __repr__(self) -> str:
        return '{!r}.{}'.format(self._ent, self._disp_attr)

    def __eq__(self, other) -> bool:
        """We're private, so we should be the only instance for a given Entity."""
        return other is self

    def _maps(self, ent=None) -> Iterator[Mapping[str, Mapping[FrozenSet[str], T]]]:
        """Yield all the mappings which we need to look through."""
        if ent is None:
            ent = self._ent

        yield getattr(ent, self._attr)
        for base in ent.bases:
            yield from self._maps(base)

    def __getitem__(self, name: Union[str, Tuple[str, Collection[str]]]) -> T:
        """Lookup the value in the entity.

        Either obj['name'], or obj['name', {tags}] is accepted.
        """
        if isinstance(name, str):
            search_tags = None
        elif isinstance(name, tuple):
            name, search_tags = name
            search_tags = frozenset({t.upper() for t in search_tags})
        else:
            raise TypeError(
                'Expected str or (str, Iterable[str]), '
                'got "{}"'.format(name),
            )
        name = name.casefold()
        for ent_map in self._maps():
            try:
                tag_map = ent_map[name]
            except KeyError:
                continue

            # Force longer more-specific tags to match first.
            for tags, value in sorted(
                tag_map.items(),
                key=lambda t: len(t[0]),
                reverse=True,
            ):
                if search_tags is None or match_tags(search_tags, tags):
                    return value
        raise KeyError((name, search_tags))

    def __iter__(self) -> Iterator[str]:
        """Yields all keys this object has."""
        seen: Set[str] = set()
        for ent_map in self._maps():
            for name in ent_map:
                if name in seen:
                    continue
                seen.add(name)
                yield name

    def __contains__(self, item: object) -> bool:
        for ent_map in self._maps():
            if item in ent_map:
                return True
        return False

    def __len__(self) -> int:
        seen: Set[str] = set()
        for ent_map in self._maps():
            seen.update(ent_map)
        return len(seen)


# Fix a bug in some typing versions - slots can't be used with generics.
del _EntityView.__slots__


@attr.define(slots=False, eq=False)
class EntityDef:
    """A definition for an entity."""
    type: EntityTypes
    classname: str = ''

    # These are (name) -> {tags} -> value dicts.
    keyvalues: Dict[str, Dict[FrozenSet[str], KeyValues]] = attr.Factory(dict)
    inputs: Dict[str, Dict[FrozenSet[str], IODef]] = attr.Factory(dict)
    outputs: Dict[str, Dict[FrozenSet[str], IODef]] = attr.Factory(dict)

    # Keyvalues have an order. If not present in here,
    # they appear at the end.
    kv_order: List[str] = attr.Factory(list)

    # Base type names - base()
    bases: List[Union['EntityDef', str]] = attr.Factory(list)
    helpers: List[Helper] = attr.Factory(list)
    desc: str = ''

    # Views for accessing data among all the entities.
    kv: _EntityView[KeyValues] = attr.ib(init=False, default=attr.Factory(
        partial(_EntityView, attr_name='keyvalues', disp_name='kv'),
        takes_self=True,
    ))
    inp: _EntityView[IODef] = attr.ib(init=False, default=attr.Factory(
        partial(_EntityView, attr_name='inputs', disp_name='inp'),
        takes_self=True,
    ))
    out: _EntityView[IODef] = attr.ib(init=False, default=attr.Factory(
        partial(_EntityView, attr_name='outputs', disp_name='out'),
        takes_self=True,
    ))

    @classmethod
    def parse(
        cls,
        fgd: 'FGD',
        tok: BaseTokenizer,
        ent_type: EntityTypes,
        eval_bases: bool=True,
    ):
        """Parse an entity definition."""
        entity = cls(ent_type)

        # First parse the bases part - lots of name(args) sections until an '='.
        ext_autovisgroups: List[List[str]] = []
        help_type: Optional[HelperTypes] = None
        help_type_cust: Optional[str] = None
        for token, token_value in tok:
            if token is Token.NEWLINE:
                continue
            if token is Token.STRING:
                if help_type is None:
                    try:
                        help_type = HelperTypes(token_value)
                    except ValueError:
                        help_type_cust = token_value
                else:
                    # No arguments for the previous helper, add it in.
                    try:
                        entity.helpers.append(HELPER_IMPL[help_type].parse([]))
                    except ValueError as exc:
                        raise tok.error(
                            'Invalid helper arguments for {}()',
                            help_type.value
                        ) from exc
                    help_type = None
                    # Then repeat this token so it's parsed.
                    tok.push_back(token, token_value)
                continue

            elif token is Token.PAREN_ARGS:
                if help_type is None and help_type_cust is None:
                    raise tok.error('Args without helper type! ({!r})', token_value)

                args = [
                    arg.strip()
                    for arg in
                    token_value.split(',')
                ]
                # helper() produces [''], when we want []
                if len(args) == 1 and args[0] == '':
                    args.clear()

                if help_type_cust is not None:
                    entity.helpers.append(UnknownHelper(help_type_cust, args))
                elif help_type is None:
                    raise tok.error('help_type not set?')
                elif help_type is HelperTypes.INHERIT:
                    for base_s in args:
                        base: Union[str, EntityDef] = fgd[base_s] if eval_bases else base_s
                        if base not in entity.bases:
                            entity.bases.append(base)
                elif help_type is HelperTypes.EXT_AUTO_VISGROUP:
                    if len(args) > 0 and args[0].casefold() != 'auto':
                        args.insert(0, 'Auto')
                    if len(args) < 2:
                        raise tok.error('autovis() requires 2 or more arguments!')
                    ext_autovisgroups.append(args)
                else:
                    try:
                        entity.helpers.append(HELPER_IMPL[help_type].parse(args))
                    except (TypeError, ValueError) as exc:
                        raise tok.error(
                            'Invalid helper arguments for {}():\n',
                            help_type.value,
                            '\n'.join(map(str, exc.args)),
                        ) from exc

                help_type = help_type_cust = None

            elif token is Token.EQUALS:
                break
            else:
                raise tok.error(token)
        else:
            raise tok.error('Entity header never ended!')

        # We were waiting for arguments for the previous helper.
        # We need to add with none.
        if help_type_cust is not None:
            entity.helpers.append(UnknownHelper(help_type_cust, []))
        elif help_type:
            if help_type is HelperTypes.EXT_AUTO_VISGROUP or help_type is HelperTypes.INHERIT:
                raise tok.error('{}() requires at least one argument!', help_type.value)
            try:
                entity.helpers.append(HELPER_IMPL[help_type].parse([]))
            except ValueError as exc:
                raise tok.error(
                    'Invalid helper arguments for {}()',
                    help_type.value
                ) from exc

        entity.classname = tok.expect(Token.STRING).strip()

        # We next might have a ':' then docstring before the [,
        # or directly to [.
        desc: Optional[List[str]] = None
        for doc_token, token_value in tok:
            if doc_token is Token.NEWLINE:
                continue
            if doc_token is Token.COLON:
                if desc is None:
                    desc = []
                else:
                    raise tok.error('Two colons in entity description!')
            elif doc_token is Token.STRING:
                if desc is None or desc:
                    # No colon yet, or we have text without '+' between
                    raise tok.error(doc_token)
                desc.append(token_value)
            elif doc_token is Token.PLUS:
                if not desc:
                    raise tok.error('+ without string before it!')
                desc.append(tok.expect(Token.STRING))
            elif doc_token is Token.BRACK_OPEN:
                if desc:
                    entity.desc = ''.join(desc)
                break
            else:
                raise tok.error(doc_token)

        fgd.entities[entity.classname.casefold()] = entity

        # Now apply EXT_AUTO_VISGROUP, since we have the classname.
        for auto_visgroup in ext_autovisgroups:
            for vis_parent, vis_name in zip(auto_visgroup, auto_visgroup[1:]):
                try:
                    visgroup = fgd.auto_visgroups[vis_name.casefold()]
                except KeyError:
                    visgroup = fgd.auto_visgroups[vis_name.casefold()] = AutoVisgroup(vis_name, vis_parent)
                visgroup.ents.add(entity.classname)

        # Now parse keyvalues, and input/outputs
        for token, token_value in tok:
            if token is Token.BRACK_CLOSE:
                break  # End of this entity.

            if token is Token.NEWLINE:
                continue

            # IO - keyword at the start.
            if token is not Token.STRING:
                raise tok.error(token)

            io_type = token_value.casefold()
            if io_type in ('input', 'output'):
                name = tok.expect(Token.STRING)

                # Next is either the value type parens, or a tags brackets.
                val_token, raw_value_type = tok()
                if val_token is Token.BRACK_OPEN:
                    tags = read_tags(tok)
                    val_token, raw_value_type = tok()
                else:
                    tags = frozenset()

                raw_value_type = raw_value_type.strip()
                if raw_value_type == 'ehandle':
                    # This is a duplicate (deprecated) name, but only for I/O.
                    val_typ = ValueTypes.EHANDLE
                else:
                    try:
                        val_typ = VALUE_TYPE_LOOKUP[raw_value_type.casefold()]
                    except KeyError:
                        raise tok.error('Unknown keyvalue type "{}"!', raw_value_type)

                if not val_typ.valid_for_io:
                    raise tok.error(
                        '"{}" value type is not valid for an input or '
                        'output! Use "{}" instead.',
                        val_typ.value,
                        VALUE_TO_IO_DECAY[val_typ].value,
                    )

                # Read desc
                io_vals, token = read_colon_list(tok)

                if token is token.EQUALS:
                    raise tok.error(token)

                if io_vals:
                    try:
                        [io_desc] = io_vals
                    except ValueError:
                        raise tok.error('Too many values for IO definition!')
                else:
                    io_desc = ''

                # entity.inputs or entity.outputs
                tags_map = getattr(entity, io_type + 's').setdefault(name.casefold(), {})
                tags_map[tags] = IODef(name, val_typ, io_desc)

            else:
                # Keyvalue
                name = io_type
                is_readonly = show_in_report = had_colon = False

                # Next is either the value type parens, or a tags brackets.

                val_token, raw_value_type = tok()
                if val_token is Token.BRACK_OPEN:
                    tags = read_tags(tok)
                    val_token, raw_value_type = tok()
                else:
                    tags = frozenset()

                if val_token is not Token.PAREN_ARGS:
                    raise tok.error(val_token)

                raw_value_type = raw_value_type.strip()
                if raw_value_type.startswith('*'):
                    # Old format for specifying 'reportable' flag.
                    show_in_report = True
                    raw_value_type = raw_value_type[1:]
                try:
                    val_typ = VALUE_TYPE_LOOKUP[raw_value_type.casefold()]
                except KeyError:
                    raise tok.error('Unknown keyvalue type "{}"!', raw_value_type)

                # Look for the 'readonly' and 'report' flags, in that order.
                next_token, key_flag = tok()
                if next_token is Token.STRING and key_flag.casefold() == 'readonly':
                    is_readonly = True
                    # Fetch next in case it has both.
                    next_token, key_flag = tok()

                if next_token is Token.STRING and key_flag.casefold() == 'report':
                    show_in_report = True
                    # Fetch for the rest of the checks.
                    next_token, key_flag = tok()

                has_equal: Optional[Token] = None
                kv_vals: Optional[List[str]] = None

                if next_token is Token.COLON:
                    had_colon = True
                elif next_token is Token.EQUALS:
                    # Special case - spawnflags doesn't have to have
                    # any info - skips straight to the end.
                    if val_typ is ValueTypes.SPAWNFLAGS:
                        kv_vals = []
                        has_equal = next_token
                elif next_token is Token.NEWLINE:
                    kv_vals = []
                    has_equal = next_token
                else:
                    raise tok.error(next_token)

                if kv_vals is None:
                    kv_vals, has_equal = read_colon_list(tok, had_colon)
                attr_len = len(kv_vals)

                kv_desc = default = ''
                if attr_len == 3:
                    disp_name, default, kv_desc = kv_vals
                elif attr_len == 2:
                    disp_name, default = kv_vals
                elif attr_len == 1:
                    [disp_name] = kv_vals
                elif attr_len == 0:
                    disp_name = name
                else:
                    raise tok.error('Too many attributes for keyvalue!\n{!r}', kv_vals)

                if val_typ is ValueTypes.BOOL:
                    # These are old aliases, change them to proper booleans.
                    if default.casefold() == 'yes':
                        default = '1'
                    elif default.casefold() == 'no':
                        default = '0'

                if val_typ.has_list:
                    if has_equal is not Token.EQUALS:
                        raise tok.error('No list for "{}" value type!', val_typ.name)
                    # Read the choices in the [].  There's two kinds of tuples here, typing this
                    # doesn't work right.
                    val_list: Optional[List[Any]] = []
                    tok.expect(Token.BRACK_OPEN)
                    for choices_token, choices_value in tok:
                        if choices_token is Token.NEWLINE:
                            continue
                        if choices_token is Token.BRACK_CLOSE:
                            break
                        elif choices_token is not Token.STRING:
                            raise tok.error(choices_token)
                        vals, end_token = read_colon_list(tok, had_colon=False)

                        if end_token is Token.BRACK_OPEN:
                            val_tags = read_tags(tok)
                        else:
                            val_tags = frozenset()

                        if val_typ is ValueTypes.SPAWNFLAGS:
                            # The first value is an integer.
                            try:
                                spawnflag = int(choices_value)
                            except ValueError:
                                raise tok.error(
                                    'SpawnFlags must be integer values, '
                                    'not "{}" (in {})!'.format(
                                        choices_value,
                                        entity.classname,
                                    )
                                ) from None
                            try:
                                power = math.log2(spawnflag)
                            except ValueError:
                                power = 0.5  # Force the following code to raise
                            if power != round(power):
                                raise tok.error(
                                    'SpawnFlags must be powers of two, '
                                    'not {} (in {})!'.format(
                                        spawnflag,
                                        entity.classname,
                                    )
                                ) from None
                            # Spawnflags can have a default, others may not.
                            if len(vals) == 2:
                                val_list.append((spawnflag, vals[0], vals[1].strip() == '1', val_tags))
                            elif len(vals) == 1:
                                val_list.append((spawnflag, vals[0], True, val_tags))
                            elif len(vals) == 0:
                                raise tok.error('Expected value for spawnflags, got none!')
                            else:
                                raise tok.error('Too many values!\n{}', vals)
                        else:  # Choices.
                            if len(vals) == 1:
                                val_list.append((choices_value, vals[0], val_tags))
                            elif len(vals) == 0:
                                raise tok.error('Expected value for choices, got none!')
                            else:
                                raise tok.error('Too many values!\n{}', vals)

                        # Handle ] at the end of a : : line.
                        if end_token is Token.BRACK_CLOSE:
                            break
                    else:
                        raise tok.error(Token.EOF)
                else:
                    val_list = None
                    if has_equal is Token.EQUALS:
                        raise tok.error('"{}" value types can\'t have lists!', val_typ.name)

                tags_map = entity.keyvalues.setdefault(name.casefold(), {})
                if not tags_map:
                    # New, add to the ordering.
                    entity.kv_order.append(name.casefold())

                tags_map[tags] = KeyValues(
                    name=name,
                    type=val_typ,
                    desc=kv_desc,
                    disp_name=disp_name,
                    default=default,
                    val_list=val_list,
                    readonly=is_readonly,
                    reportable=show_in_report,
                )

    def __repr__(self) -> str:
        if self.type is EntityTypes.BASE:
            return '<Entity Base "{}">'.format(self.classname)
        else:
            return '<Entity {}>'.format(self.classname)

    def __deepcopy__(self, memodict: dict) -> 'EntityDef':
        """Handle copying ourselves, to eliminate lookups when not required."""
        copy = EntityDef.__new__(EntityDef)
        copy.type = self.type
        copy.classname = self.classname
        copy.kv_order = self.kv_order.copy()
        copy.bases = deepcopy(self.bases, memodict)
        copy.helpers = deepcopy(self.helpers, memodict)
        copy.desc = self.desc

        # Avoid copy for these, we know the tags-map is immutable.
        for val_key in ['keyvalues', 'inputs', 'outputs']:
            coll: Dict[str, Dict[FrozenSet[str], Union[KeyValues, IODef]]] = {}
            setattr(copy, val_key, coll)
            tags_map: Dict[FrozenSet[str], Union[KeyValues, IODef]]
            for key, tags_map in getattr(self, val_key).items():
                coll[key] = {
                    key: value.copy()
                    for key, value in tags_map.items()
                }
        copy.kv = _EntityView(copy, 'keyvalues', 'kv')
        copy.inp = _EntityView(copy, 'inputs', 'inp')
        copy.out = _EntityView(copy, 'outputs', 'out')
        return copy

    def __getstate__(self) -> tuple:
        """Don't include EntityView while pickling."""
        return (
            self.type,
            self.classname,
            self.keyvalues,
            self.inputs,
            self.outputs,
            self.kv_order,
            self.bases,
            self.helpers,
            self.desc
        )

    def __setstate__(self, state: tuple) -> None:
        """We can regenerate EntityView from scratch."""
        (
            self.type,
            self.classname,
            self.keyvalues,
            self.inputs,
            self.outputs,
            self.kv_order,
            self.bases,
            self.helpers,
            self.desc
        ) = state
        self.kv = _EntityView(self, 'keyvalues', 'kv')
        self.inp = _EntityView(self, 'inputs', 'inp')
        self.out = _EntityView(self, 'outputs', 'out')

    @overload
    def get_helpers(self, typ: Type[HelperT]) -> Iterator[HelperT]: ...
    @overload
    def get_helpers(self, typ: str) -> Iterator[UnknownHelper]: ...

    def get_helpers(self, typ: Union[Type[HelperT], str]) -> Iterator[Helper]:
        """Find all helpers with this specific type."""
        if isinstance(typ, str):
            for helper in self.helpers:
                if isinstance(helper, UnknownHelper) and helper.name == typ:
                    yield helper
        else:
            for helper in self.helpers:
                if helper.TYPE == typ.TYPE:
                    yield helper

    def _iter_attrs(self) -> Iterator[Dict[str, Dict[FrozenSet[str], Union[KeyValues, IODef]]]]:
        """Iterate over both the keyvalues and I/O dicts.

        This is used when we want to deal with both in the same way.
        """
        return iter([self.keyvalues, self.inputs, self.outputs])  # type: ignore

    def strip_tags(self, tags: FrozenSet[str]) -> None:
        """Strip all tags from this entity, blanking them.

        Only values matching the given tags will be kept.
        """
        for category in self._iter_attrs():
            for key, tag_map in list(category.items()):
                # Force longer more-specific tags to match first.
                for key_tag, value in sorted(
                    tag_map.items(),
                    key=lambda t: len(t[0]),
                    reverse=True,
                ):
                    if match_tags(tags, key_tag):
                        category[key] = {frozenset(): value}
                        if isinstance(value, KeyValues) and value.val_list:
                            # Filter the value list as well, then discard tags.
                            value.val_list = [  # type: ignore
                                val[:-1] + (frozenset(), )
                                for val in value.val_list
                                if match_tags(tags, val[-1])
                            ]
                        break
                else:
                    del category[key]

    def export(self, file: TextIO) -> None:
        """Write the entity out to a FGD file."""
        # Make it look pretty: BaseClass
        file.write('@{} '.format(
            self.type.value.title().replace('class', 'Class')
        ))
        if self.bases:
            file.write('base(')
            file.write(', '.join([
                (base.classname if isinstance(base, EntityDef) else base)
                for base in self.bases
            ]))
            file.write(') ')

        kv_order_list: List[str] = []

        for helper in self.helpers:
            args = helper.export()
            if isinstance(helper, HelperHalfGridSnap):
                # Special case, no args.
                file.write('\n\thalfgridsnap')
            elif isinstance(helper, UnknownHelper):
                file.write('\n\t{}({})'.format(helper.name, ', '.join(args)))
            elif helper.TYPE is not None:
                file.write('\n\t{}({})'.format(helper.TYPE.value, ', '.join(args)))
            else:
                raise TypeError(f'Helper {helper!r} has no TYPE attr?')
            if isinstance(helper, HelperExtOrderBy):
                kv_order_list.extend(map(str.casefold, args))

        if self.helpers:
            file.write('\n')  # Put the classname on the following line.
        file.write('= {}'.format(self.classname))

        if self.desc:
            file.write(': ')
            _write_longstring(file, self.desc, indent='\t\t')

        file.write('\n\t[\n')

        kv_order = {
            name: ind
            for ind, name in
            enumerate(kv_order_list or self.kv_order)
        }

        for name, kv_map in sorted(
            self.keyvalues.items(),
            # Sort by position in kv_order. If not present add to the end.
            key=lambda name_kv: kv_order.get(name_kv[0], 2**64),
        ):
            for tags, kv in kv_map.items():
                kv.export(file, tags)

        if self.inputs:
            file.write('\n\t// Inputs\n')

            for inp_map in self.inputs.values():
                for tags, inp in inp_map.items():
                    inp.export(file, 'input', tags)

        if self.outputs:
            file.write('\n\t// Outputs\n')

            for out_map in self.outputs.values():
                for tags, out in out_map.items():
                    out.export(file, 'output', tags)
        file.write('\t]\n')

    def iter_bases(self, _done: Set['EntityDef']=None) -> Iterator['EntityDef']:
        """Yield all base entities for this one.

        If an entity is repeated, it will only be yielded once.
        """
        if not _done:
            _done = {self}
        for ent in self.bases:
            if ent in _done or isinstance(ent, str):
                continue

            _done.add(ent)
            yield ent
            yield from ent.iter_bases(_done)

    def serialise(self, file, str_dict: BinStrDict):
        """Write to the binary file."""
        file.write(_fmt_ent_header.pack(
            ENTITY_TYPE_INDEX[self.type],
            len(self.bases),
            len(self.keyvalues),
            len(self.inputs),
            len(self.outputs),
            # Write the classname here, not using BinStrDict.
            # They're going to be unique, so there's no benefit.
            len(self.classname),
        ))
        file.write(self.classname.encode())

        for base_ent in self.bases:
            if isinstance(base_ent, str):
                file.write(str_dict(base_ent))
            else:
                file.write(str_dict(base_ent.classname))

        for obj_type in self._iter_attrs():
            for tag_map in obj_type.values():
                # We don't need to write the name, since that's stored
                # also in the kv/io object itself.

                if not tag_map:
                    # No need to add this one.
                    continue

                # Special case - if there is one blank tag, write len=0
                # and just the value.
                # That saves 2 bytes.
                if len(tag_map) == 1:
                    [(tags, value)] = tag_map.items()
                    if not tags:
                        file.write(_fmt_8bit.pack(0))
                        value.serialise(file, str_dict)
                        continue

                file.write(_fmt_8bit.pack(len(tag_map)))
                for tags, value in tag_map.items():
                    BinStrDict.write_tags(file, str_dict, tags)
                    value.serialise(file, str_dict)

        # Helpers are not added.

    @staticmethod
    def unserialise(
        file: IO[bytes],
        from_dict: Callable[[], str],
    ) -> 'EntityDef':
        """Read from the binary file."""
        [
            type_ind,
            base_count,
            kv_count,
            inp_count,
            out_count,
            clsname_length,
        ] = struct_read(_fmt_ent_header, file)

        ent = EntityDef(ENTITY_TYPE_ORDER[type_ind])
        ent.classname = file.read(clsname_length).decode('utf8')
        ent.desc = ''

        for _ in range(base_count):
            # We temporarily store strings, then evaluate later on.
            ent.bases.append(from_dict())  # type: ignore

        count: int
        val_map: Dict[str, Dict[FrozenSet[str], Union[KeyValues, IODef]]]
        cls: Type[Union[KeyValues, IODef]]
        for count, val_map, cls in [  # type: ignore
            (kv_count, ent.keyvalues, KeyValues),
            (inp_count, ent.inputs, IODef),
            (out_count, ent.outputs, IODef),
        ]:
            for _ in range(count):
                [tag_count] = struct_read(_fmt_8bit, file)
                if tag_count == 0:
                    # Special case, a single untagged item.
                    obj = cls.unserialise(file, from_dict)
                    val_map[obj.name] = {frozenset(): obj}
                else:

                    # We know it's starting empty, and must have at least
                    # one tag.

                    tag = BinStrDict.read_tags(file, from_dict)
                    obj = cls.unserialise(file, from_dict)
                    tag_map = val_map[obj.name] = {tag: obj}
                    for _ in range(tag_count - 1):
                        tag = BinStrDict.read_tags(file, from_dict)
                        obj = cls.unserialise(file, from_dict)
                        tag_map[tag] = obj

        return ent


class FGD:
    """A FGD set for a game. May be composed of several files."""
    # List of names we have already parsed.
    # We don't parse them again, to prevent infinite loops.
    _parse_list: Set[File]
    # Entity definitions
    entities: Dict[str, EntityDef]
    # Maximum bounding box of map
    map_size_min: int
    map_size_max: int

    # Directories we have excluded.
    mat_exclusions: Set[PurePosixPath]
    # Additional dirs restricted to specific engines with tags.
    tagged_mat_exclusions: Dict[FrozenSet[str], Set[PurePosixPath]]

    # Automatic visgroups.
    # The way Valve implemented this is rather strange, so we need to match
    # their data structure really to get good results. Despite it appearing
    # hierarchical in editor, we and Hammer store it flattened. Each visgroup
    # has a parent (or None for auto), and then a list of the ents it contains.
    auto_visgroups: Dict[str, AutoVisgroup]
    def __init__(self) -> None:
        """Create a FGD."""
        self._parse_list = set()
        self.entities = {}
        self.map_size_min = self.map_size_max = 0
        self.mat_exclusions = set()
        self.tagged_mat_exclusions = defaultdict(set)
        self.auto_visgroups = {}

    @classmethod
    def parse(
        cls,
        file: Union[File, str],
        filesystem: FileSystem=None,
    ) -> 'FGD':
        """Parse an FGD file.

        Parameters:
        * file: A filesys.File representing the file to read, or a file path.
        * filesystem: The system to lookup files in. This is needed to
          resolve file inclusions. If not passed, file must be a filesystem
          File to obtain a matching filesystem.
        """
        if filesystem is not None and not isinstance(file, File):
            if not file.endswith('.fgd'):
                file += '.fgd'
            try:
                with filesystem:
                    file = filesystem[file]
            except KeyError:
                raise FileNotFoundError(file)
        elif isinstance(file, File):
            filesystem = file.sys
        else:
            raise TypeError(
                'String file path passed ({!r}), but no filesystem!'.format(file)
            )
        assert filesystem is not None, (filesystem, file)
        fgd = cls()
        fgd.parse_file(filesystem, file)
        return fgd

    def apply_bases(self) -> None:
        """Fix base values in entities after parsing.

        While parsing the classnames may be set as strings,
        so order in the file doesn't matter. This fixes
        them to the real entity objects.
        """
        for ent in self:
            orig_bases = ent.bases
            new_bases = ent.bases = []
            for base in orig_bases:
                if isinstance(base, EntityDef):
                    # This entity was already done.
                    new_bases.append(base)
                    continue

                try:
                    new_bases.append(self[base])  # type: ignore
                except KeyError:
                    raise ValueError(
                        'Unknown base ({}) for {}'.format(
                            base,
                            ent.classname,
                        )
                    ) from None

    def sorted_ents(self) -> Iterator[EntityDef]:
        """Yield all entities in sorted order.

        This ensures only all bases for an entity are yielded before the entity.
        Otherwise entities are ordered in alphabetical order.
        """
        # We need to do a topological sort.
        todo: Set[EntityDef] = set(self)
        done: Set[EntityDef] = set()
        cls_getter = operator.attrgetter('classname')
        while todo:
            deferred: Set[EntityDef] = set()
            batch = []
            for ent in todo:
                ready = True
                for base in ent.bases:
                    if isinstance(base, str):
                        raise ValueError(
                            'Unevaluated base: {} in {}!'.format(
                                base, ent.classname,
                            ))
                    if base not in done:
                        # Base not done yet, we need to defer this.
                        deferred.add(ent)
                        # If the base isn't in any of our sets, it's one
                        # just in the .bases attr, not in the fgd.entities
                        # dict - defer it too so it can be added.
                        if base not in todo:
                            deferred.add(base)
                        ready = False
                if ready:
                    batch.append(ent)
                else:
                    deferred.add(ent)

            batch.sort(key=cls_getter)
            yield from batch

            done.update(batch)

            # All the entities have a dependency on another, we failed to produce anything.
            if not batch:
                raise ValueError(
                    "Loop in bases! \n Problematic entities: \n{}".format([
                        ent.classname
                        for ent in deferred
                    ]))

            todo = deferred.difference(done)

    def collapse_bases(self) -> None:
        """Collapse all bases into the entities that use them.

        This operates in-place, and clears all the base attributes as a result.
        """
        # We need to do a topological sort effectively, to ensure we do
        # parents before children.
        for ent in self.sorted_ents():
            base_kv = []
            keyvalue_names = set(ent.kv_order)

            for base in ent.bases:
                if isinstance(base, str):
                    try:
                        base = self[base]
                    except KeyError:
                        raise ValueError(
                            'Unknown base ({}) for {}'.format(
                                base,
                                ent.classname,
                            )
                        ) from None
                for name, base_kv_map in base.keyvalues.items():
                    ent_kv_map = ent.keyvalues.setdefault(name, {})
                    for tag, kv in base_kv_map.items():
                        if tag not in ent_kv_map:
                            ent_kv_map[tag] = kv.copy()
                        elif kv.type.has_list and ent_kv_map[tag].type is kv.type:
                            # If both are lists of the same type, merge those. This is mainly
                            # for spawnflags.
                            targ_list = ent_kv_map[tag].val_list
                            if targ_list is not None and kv.val_list is not None:
                                for val in kv.val_list:
                                    if val not in targ_list:
                                        # Type checker can't know type attr indicates val_list type.
                                        targ_list.append(val)  # type: ignore

                    if name not in keyvalue_names:
                        base_kv.append(name)
                        keyvalue_names.add(name)

                for base_map, ent_map in [
                    (base.inputs, ent.inputs),
                    (base.outputs, ent.outputs),
                ]:
                    for name, base_tags_map in base_map.items():
                        ent_tags_map = ent_map.setdefault(name, {})
                        for tag, io_def in base_tags_map.items():
                            if tag not in ent_tags_map:
                                ent_tags_map[tag] = io_def.copy()

            ent.kv_order = base_kv + ent.kv_order
            ent.bases.clear()

    @overload
    def export(self, file: TextIO) -> None: ...
    @overload
    def export(self) -> str: ...
    def export(self, file=None):
        """Write the FGD contents into a text file.

        If none are provided, the text will be returned.
        """
        if file is None:
            file = io.StringIO()
            ret_string = True
        else:
            ret_string = False

        if self.map_size_min != self.map_size_max:
            file.write('@mapsize({}, {})\n\n'.format(self.map_size_min, self.map_size_max))

        if self.mat_exclusions:
            file.write('@MaterialExclusion\n\t[\n')
            for folder in sorted(self.mat_exclusions):
                file.write('\t"{!s}"\n'.format(folder))
            file.write('\t]\n\n')
        for tag in sorted(self.tagged_mat_exclusions):
            file.write(f'@MaterialExclusion({", ".join(sorted(tag))})\n\t[\n')
            for folder in sorted(self.tagged_mat_exclusions[tag]):
                file.write('\t"{!s}"\n'.format(folder))
            file.write('\t]\n\n')

        vis_by_parent: dict[str, set[AutoVisgroup]] = defaultdict(set)
        # Record the proper casing as well.
        name_casing = {'auto': 'Auto'}
        for visgroup in list(self.auto_visgroups.values()):
            if not visgroup.parent:
                visgroup.parent = 'Auto'
            elif visgroup.parent.casefold() not in self.auto_visgroups:
                # This is an "orphan" visgroup, not linked back to Auto.
                # Connect it back there, by generating the parent.
                parent_group = self.auto_visgroups[visgroup.parent.casefold()] = AutoVisgroup(visgroup.parent, 'Auto')
                vis_by_parent['auto'].add(parent_group)
                parent_group.ents.update(visgroup.ents)
            vis_by_parent[visgroup.parent.casefold()].add(visgroup)
            name_casing[visgroup.parent.casefold()] = visgroup.parent

        # We need to sort these, so we write parents before children.
        todo = set(vis_by_parent)
        done = set()
        while todo:
            deferred = set()
            for parent in sorted(todo):
                # Special case the root, pretend that was written to the file.
                if parent != 'auto':
                    visgroup = self.auto_visgroups[parent.casefold()]
                    if visgroup.parent.casefold() not in done:
                        deferred.add(parent)
                        continue
                # Otherwise, the parent is done, so we can generate.
                file.write('@AutoVisgroup = "{}"\n\t[\n'.format(name_casing[parent]))
                for visgroup in sorted(vis_by_parent[parent]):
                    file.write('\t"{}"\n\t\t[\n'.format(visgroup.name))
                    for ent in sorted(visgroup.ents):
                        file.write('\t\t"{}"\n'.format(ent))
                    file.write('\t\t]\n')
                file.write('\t]\n')
                done.add(parent)

            if todo == deferred:
                # We looped without adding one. There's an invalid one or
                # a loop or something.
                raise ValueError(
                    'Cannot export visgroups, '
                    'loop present in names: {}'.format(','.join([
                        '"{}" -> "{}"'.format(self.auto_visgroups[group].parent, group)
                        for group in sorted(todo)
                    ]))
                )
            todo = deferred

        for ent_def in self.sorted_ents():
            file.write('\n')
            ent_def.export(file)

        if ret_string:
            return file.getvalue()

    def parse_file(
        self,
        filesys: FileSystem,
        file: File,
        *,
        eval_bases: bool=True,
        encoding='cp1252',
    ) -> None:
        """Parse one file (recursively if needed).

        If eval_bases is False, bases will not be computed. This makes it
        impossible in some cases to evaluate these later, but it can help
        if it is not required.
        """

        if file in self._parse_list:
            return

        self._parse_list.add(file)

        with file.open_str(encoding) as f:
            tokeniser = Tokenizer(
                f,
                filename=file.path,
                error=FGDParseError,
                string_bracket=False,
                colon_operator=True,
            )
            for token, token_value in tokeniser:
                # The only things at top-level would be bare strings, and empty lines.
                if token is Token.NEWLINE:
                    continue
                if token is not Token.STRING:
                    raise tokeniser.error(token)
                token_value = token_value.casefold()

                if token_value == '@include':
                    include_file = tokeniser.expect(Token.STRING)
                    if not include_file.endswith('.fgd'):
                        include_file += '.fgd'

                    try:
                        include = filesys[include_file]
                    except KeyError:
                        raise FileNotFoundError(file)
                    self.parse_file(
                        filesys,
                        include,
                        eval_bases=eval_bases,
                        encoding=encoding,
                    )

                elif token_value == '@mapsize':
                    # Max/min map size definition
                    mapsize_args = tokeniser.expect(Token.PAREN_ARGS)
                    try:
                        min_size, max_size = mapsize_args.split(',')
                        self.map_size_min = int(min_size.strip())
                        self.map_size_max = int(max_size.strip())
                    except ValueError:
                        raise tokeniser.error(
                            'Invalid @MapSize: ({})',
                            mapsize_args,
                        )
                elif token_value == '@materialexclusion':
                    # Material exclusion directories.

                    # Custom syntax: (tag1, tag2) after the header, then [.
                    tags: Optional[FrozenSet[str]] = None
                    for tok, tok_value in tokeniser.skipping_newlines():
                        if tok is Token.BRACK_OPEN:
                            break
                        elif tok is Token.PAREN_ARGS and tags is None:
                            tags = validate_tags(tok_value.split(','), tokeniser.error)
                        else:
                            raise tokeniser.error(tok)

                    for tok, tok_value in tokeniser:
                        if tok is Token.BRACK_CLOSE:
                            break
                        elif tok is Token.STRING:
                            if tags is not None:
                                self.tagged_mat_exclusions[tags].add(PurePosixPath(tok_value))
                            else:
                                self.mat_exclusions.add(PurePosixPath(tok_value))
                        elif tok is not Token.NEWLINE:
                            raise tokeniser.error(tok)
                    else:
                        raise tokeniser.error(
                            'Missing closing bracket '
                            'for @MaterialExclusion!'
                        )

                elif token_value == '@autovisgroup':
                    tokeniser.expect(Token.EQUALS)
                    vis_parent = tokeniser.expect(Token.STRING)
                    tokeniser.expect(Token.BRACK_OPEN)

                    for tok, vis_name in tokeniser:
                        if tok is Token.BRACK_CLOSE:
                            break
                        elif tok is Token.STRING:
                            # Folder
                            try:
                                visgroup = self.auto_visgroups[vis_name.casefold()]
                            except KeyError:
                                visgroup = self.auto_visgroups[vis_name.casefold()] = AutoVisgroup(vis_name, vis_parent)

                            tokeniser.expect(Token.BRACK_OPEN)
                            for ent_tok, ent_tok_value in tokeniser:
                                if ent_tok is Token.BRACK_CLOSE:
                                    break
                                elif ent_tok is Token.STRING:
                                    # Entity
                                    visgroup.ents.add(ent_tok_value)
                                elif ent_tok is not Token.NEWLINE:
                                    raise tokeniser.error(ent_tok)
                        elif tok is not Token.NEWLINE:
                            raise tokeniser.error(tok)

                # Entity definition...
                elif token_value[:1] == '@':
                    try:
                        ent_type = EntityTypes(token_value[1:])
                    except ValueError:
                        raise tokeniser.error(
                            'Invalid Entity type "{}"!',
                            token_value[1:],
                        )
                    EntityDef.parse(self, tokeniser, ent_type, eval_bases)
                else:
                    raise tokeniser.error('Bad keyword {!r}', token_value)

    @classmethod
    def engine_dbase(cls) -> 'FGD':
        """Load and return a database of entity keyvalues and I/O.

        This can be used to identify the kind of keys present on an entity.
        """
        # It's pretty expensive to parse, so keep the original privately,
        # returning a deep-copy.
        global _ENGINE_FGD
        if _ENGINE_FGD is None:
            from lzma import LZMAFile
            with open_binary(srctools, 'fgd.lzma') as comp, LZMAFile(comp) as f:
                return cls.unserialise(f)
        return deepcopy(_ENGINE_FGD)

    def __getitem__(self, classname: str) -> EntityDef:
        """Lookup entities by classname."""
        try:
            return self.entities[classname.casefold()]
        except KeyError:
            raise KeyError('No class "{}"!'.format(classname)) from None

    def __contains__(self, classname: object) -> bool:
        """Lookup entities by classname."""
        if isinstance(classname, str):
            return classname.casefold() in self.entities
        return False

    def __iter__(self) -> Iterator[EntityDef]:
        """Iterating over FGDs iterates over the entities."""
        return iter(self.entities.values())

    def __len__(self) -> int:
        """The length of the FGD is the number of entities."""
        return len(self.entities)

    def _fix_missing_bases(self, ent: EntityDef) -> None:
        """Fix issues that prevent serialising base entities.

        The FGD implementation by Valve is very order-dependent.
        It is possible to have a base class overwritten by a real entity,
        as long as it comes before that in the file. To allow serialising
        this, fix those entities by appending numbers to any not in the list.
        This is run recursively on every entity to check their bases.
        """
        for base in ent.bases:
            if isinstance(base, str):
                continue
            # If it's in there, it'll be found again.
            # We've also (or are going to) pass over this one.
            # So don't redo it.
            if self[base.classname] is base:
                continue

            base_name = base.classname.rstrip('_0123456789').casefold() + '_'
            for num in itertools.count(1):
                poss_name = base_name + str(num)
                if poss_name not in self.entities:
                    base.classname = poss_name
                    self.entities[poss_name] = base
                    break
            self._fix_missing_bases(base)

    def serialise(self, file: IO[bytes]) -> None:
        """Write the FGD into a compacted binary format.

        This is only readable by this module, and does not contain
        entity, keyvalue and IO help descriptions to keep the data small.
        """
        for ent in list(self):
            self._fix_missing_bases(ent)

        # The start of a file is a list of all used strings.
        dictionary = BinStrDict()

        # Start of file - format version, FGD min/max, number of entities.
        file.write(b'FGD' + _fmt_header.pack(
            BIN_FORMAT_VERSION,
            self.map_size_min,
            self.map_size_max,
            len(self.entities),
        ))

        ent_data = io.BytesIO()
        for ent in self.entities.values():
            ent.serialise(ent_data, dictionary)

        # The final file is the header, dictionary data, and all the entities
        # one after each other.
        dictionary.serialise(file)
        file.write(ent_data.getvalue())
        # print('Dict size: ', format(dictionary.cur_index / (1 << 16), '%'))

    @classmethod
    def unserialise(cls, file: IO[bytes]) -> 'FGD':
        """Unpack data from FGD.serialise() to return the original data.

        Help descriptions are not preserved, and are set to <BINARY>.
        """

        if file.read(3) != b'FGD':
            raise ValueError('Not an FGD file!')

        fgd = FGD()

        [
            format_version,
            fgd.map_size_min,
            fgd.map_size_max,
            ent_count,
        ] = struct_read(_fmt_header, file)

        if format_version != BIN_FORMAT_VERSION:
            raise TypeError('Unknown format version "{}"!'.format(format_version))

        from_dict = BinStrDict.unserialise(file)

        # Now there's ent_count entities after each other.
        for _ in range(ent_count):
            ent = EntityDef.unserialise(file, from_dict)
            fgd.entities[ent.classname.casefold()] = ent

        fgd.apply_bases()

        return fgd
