"""Parse FGD files, used to describe Hammer entities."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Optional, TypeVar, Union, cast
from typing_extensions import Protocol, TypeAlias, overload, Self
from collections import ChainMap, defaultdict
from collections.abc import (
    Callable, Collection, Container, Iterable, Iterator, Mapping, Sequence, Set as AbstractSet,
)
from copy import deepcopy
from enum import Enum
from pathlib import Path, PurePosixPath
import builtins  # type shadowed.
import io
import itertools
import math
import operator
import sys
import warnings

import attrs

from srctools.const import FileType
from srctools.filesys import File, FileSystem, VirtualFileSystem
from .tokenizer import BaseTokenizer, Token, Tokenizer, TokenSyntaxError, escape_text
from .vmf import VMF, Entity
from .types import FileWText
import srctools


__all__ = [
    'ValueTypes', 'EntityTypes', 'HelperTypes', 'AutoVisgroup', 'FGDParseError',
    'FGD', 'EntityDef', 'KVDef', 'IODef', 'EntAttribute', 'Helper', 'UnknownHelper',
    'match_tags', 'validate_tags', 'Resource', 'ResourceCtx', 'Snippet',

    # From srctools._fgd_helpers
    'HelperAxis', 'HelperBBox', 'HelperBoundingBox', 'HelperBreakableSurf',
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
    'HelperExtCatapult',
]

T = TypeVar('T')
ValueT_co = TypeVar('ValueT_co', covariant=True)
FileSysT = TypeVar('FileSysT', bound=FileSystem)
_fake_vmf = VMF(preserve_ids=False)
# Collections of tags.
TagsSet: TypeAlias = frozenset[str]
SnippetDict: TypeAlias = dict[str, list['Snippet[T]']]
SnippetLookupDict: TypeAlias = Mapping[str, Sequence['Snippet[T]']]
SpawnFlags: TypeAlias = tuple[int, str, bool, TagsSet]
Choices: TypeAlias = tuple[str, str, TagsSet]

# Cached engine DB parsing functions.
_ENGINE_DB: Optional[list[_EngineDBProto]] = None


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

    TARG_DEST_CLASS = 'target_name_or_class'  # A targetname for another ent, or a classname.
    TARG_SOURCE = 'target_source'  # The 'targetname' keyvalue.
    TARG_NPC_CLASS = 'npcclass'  # Targetnames filtered to NPC entities.
    TARG_POINT_CLASS = 'pointentityclass'  # Targetnames filtered to point entities.
    TARG_FILTER_NAME = 'filterclass'  # Targetnames filtered to filter entities.
    TARG_NODE_DEST = 'node_dest'  # Name of a node
    TARG_NODE_SOURCE = 'node_id'  # name of us.

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
    EXT_SOUNDSCAPE = 'soundscape'  # Soundscape entry. Supported natively by Strata, but not by prior branches

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


VALUE_TYPE_LOOKUP: dict[str, ValueTypes] = {
    typ.value: typ
    for typ in ValueTypes
}
# These have two names pointing to the same type...
VALUE_TYPE_LOOKUP['bool'] = ValueTypes.BOOL
VALUE_TYPE_LOOKUP['int'] = ValueTypes.INT

# In I/O definitions, types are constrained.
# So this is the most appropriate type to use instead for all types.
# First, string can reproduce everything if not allowed.
VALUE_TO_IO_DECAY: dict[ValueTypes, ValueTypes] = {
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


RESTYPE_BY_NAME = {
    'file': FileType.GENERIC,
    'entity': FileType.ENTITY,
    'func': FileType.ENTCLASS_FUNC,
    'sound': FileType.GAME_SOUND,
    'particle': FileType.PARTICLE,
    'vscript_squirrel': FileType.VSCRIPT_SQUIRREL,
    'material': FileType.MATERIAL,
    'mat': FileType.MATERIAL,
    'texture': FileType.TEXTURE,
    'choreo': FileType.CHOREO,
    'scene': FileType.CHOREO,
    'model': FileType.MODEL,

    'snd': FileType.GAME_SOUND,
    'tex': FileType.TEXTURE,
    'mdl': FileType.MODEL,
    'break_chunk': FileType.BREAKABLE_CHUNK,
    'weapon_script': FileType.WEAPON_SCRIPT,
}
RESTYPE_TO_NAME = {
    restype: name
    for name, restype in RESTYPE_BY_NAME.items()
}


class EntityTypes(Enum):
    """The kind of entity each definition is."""
    BASE = 'baseclass'  #: Not an entity, others inherit from this.
    POINT = 'pointclass'  #: Point entity
    BRUSH = 'solidclass'  #: Brush entity. Can't have a ``model`` keyvalue.
    ROPES = 'keyframeclass'  #: Used for ``move_rope`` etc
    TRACK = 'moveclass'  #: Used for ``path_track`` etc
    FILTER = 'filterclass'  #: Used for filters.
    NPC = 'npcclass'  #: An NPC.
    EXTEND = 'extendclass'  #: Modifies an existing entity entry (Hammer++ extension)

    @property
    def is_point(self) -> bool:
        """Return whether this is a point entity."""
        return self.value not in ['baseclass', 'solidclass', 'extendclass']


class HelperTypes(Enum):
    """Types of functions in the entity header."""
    INHERIT = 'base'

    # Snap to 1/2 of grid.
    # Special - no arguments.
    HALF_GRID_SNAP = 'halfgridsnap'

    # Simple helpers
    CUBE = 'size'  #: Sets bounding box of entity, exposed as the size of purple cube
    TINT = 'color'
    SPHERE = 'sphere'
    LINE = 'line'
    FRUSTUM = 'frustum'
    CYLINDER = 'cylinder'
    ORIGIN = 'origin'  #: Adds circle at an absolute position.
    VECLINE = 'vecline'  #: Draws line to an absolute position.
    AXIS = 'axis'  #: Draws a pair of lines to an absolute position.
    BRUSH_SIDES = 'sidelist'  #: Highlights brush faces.
    BOUNDING_BOX_HELPER = 'wirebox'  #: Displays bounding box from two keyvalues
    #: Draws the movement of a player-sized bounding box from A to B.
    SWEPT_HULL = 'sweptplayerhull'
    ORIENTED_BBOX = 'obb'  #: Bounding box oriented to angles.

    # Complex helpers using resources
    SPRITE = 'iconsprite'
    MODEL = 'studio'
    MODEL_PROP = 'studioprop'
    MODEL_NEG_PITCH = 'lightprop'  #: Uses separate pitch keyvalue

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
    ENT_BREAKABLE_SURF = 'quadbounds'  #: Sets the 4 corners on save
    ENT_WORLDTEXT = 'worldtext'  #: Renders 3D text in-world.
    ENT_CATAPULT = 'catapult'  #: Renders trigger_catpault trajectors prediction

    ENT_LIGHT_CONE_BLACK_MESA = 'lightconenew'  #: New helper added in Black Mesa.

    # Format extensions.

    #: Indicates this entity is only available in the given games.
    EXT_APPLIES_TO = 'appliesto'
    EXT_ORDERBY = 'orderby'  #: Reorder keyvalues. Args = names in order.

    #: Convenience only used in parsing, adds @AutoVisgroup parents for the
    #: current entity. 'Auto' is implied at the start.
    EXT_AUTO_VISGROUP = 'autovis'

    BBOX = 'bbox'  #: Deprecated, doesn't exist.

    # Additionally, aliasof(base) is used to indicate this is an alternate classname for a
    # single base.

    @property
    def extension(self) -> bool:
        """Is this an extension to the format?"""
        return self.name.startswith('EXT_')


def add_engine_database(path: Path) -> None:
    """Add an additional binary database. This can override the existing entities."""
    db = _load_engine_db()  # Ensure the first database is initialised and loaded

    with path.open('rb') as f:
        from ._engine_db import unserialise
        db.insert(0, unserialise(f))


def _load_engine_db() -> list[_EngineDBProto]:
    """Load the builtin database if required.

    This returns the resolved ``_ENGINE_DB`` value, allowing callers to avoid the `None` check.
    """
    # It's pretty expensive to parse, so keep the original privately,
    # returning a deep-copy.
    global _ENGINE_DB
    if _ENGINE_DB is None:
        from ._engine_db import unserialise
        from importlib.resources import files

        with (files(srctools) / 'fgd.lzma').open('rb') as f:
            # Assign after, so if it fails to parse every call will repeat the failure.
            _ENGINE_DB = [unserialise(f)]

    return _ENGINE_DB


def _engine_db_stats() -> str:
    """Return information about how much of the engine database has been parsed."""
    if _ENGINE_DB is None:
        return '<not loaded>'
    else:
        i = 0
        to_return = ""
        for db in _ENGINE_DB:
            to_return += f"[Database {i}]: "
            to_return += db.stats() + "\n"
            i += 1

        return to_return


def _read_colon_list(
    tok: BaseTokenizer,
    had_colon: bool = False,
    desc_offset: int = -1,
    snippet_desc: SnippetLookupDict[str] = srctools.EmptyMapping,
) -> list[str]:
    """Read strings seperated by colons, up to the end of the line.

    If desc_offset is provided, this position in the list can be a @snippet reference.
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
        elif token is Token.DIRECTIVE and tok_value == 'snippet':
            # Allow a @snippet directive to fetch a description, but only in that spot.
            if len(strings) != desc_offset:
                raise tok.error('@snippet may only be used for the description value.')
            desc_key = tok.expect(Token.STRING)

            if not ready_for_string:
                raise tok.error('Too many strings (#snippet "{}")!', desc_key)
            strings.append(Snippet.lookup_single(tok.error, 'description', snippet_desc, desc_key))
            ready_for_string = False
        elif token is Token.NEWLINE:
            if ready_for_string:
                continue  # Last line ended with +, skip the newline.
            else:
                # Check if the next line's token is a +, if so allow a string.
                line_tok, line_value = tok()
                if line_tok is Token.PLUS:
                    if not strings:
                        raise tok.error('"+" without a string before it!')
                    strings[-1] += tok.expect(Token.STRING)
                    continue
                # Put the tokens back.
                tok.push_back(line_tok, line_value)
                tok.push_back(token, tok_value)
                return strings
        else:
            if ready_for_string:
                raise tok.error(token)
            tok.push_back(token, tok_value)
            return strings
    raise tok.error(token)


def _fgd_escape(extended: bool, text: str) -> str:
    """Escape text for FGD files. The original parser only supported parsing \\n,
    but others support more escapes.

    Swap double quotes for double single quotes, which at least look somewhat right.
    """
    if extended:
        return escape_text(text)
    return text.replace('\n', '\\n').replace('"', "''")


def _write_longstring(file: FileWText, extended: bool, text: str, *, indent: str) -> None:
    """Write potentially long strings to the file, splitting with + if needed.

    The game parser has a max size of 8192 bytes for the text, but can only
    handle 1024 bytes of text in a string. So we need to split after that.
    """
    LIMIT = 1000  # Give a bit of extra room for the quotes, etc.
    sections = []
    remaining = _fgd_escape(extended, text)
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


def _parse_colon_array(
    tok: BaseTokenizer, error_desc: str, kind: str,
    snippet_mapping: SnippetLookupDict[Sequence[T]],
    parse: Callable[[BaseTokenizer, str, str, list[str], TagsSet], T],
) -> list[T]:
    """Parse through an array of colon-separated values, like in choices/flags keyvalues.

    The function provided is used to parse each line into the desired object.
    """
    tok_typ, tok_value = tok()
    if tok_typ is Token.DIRECTIVE and tok_value == "snippet":
        # A line like "... = #snippet" - include a single array, no additional values.
        # Make sure to copy the list, we don't want to share. We know the contents are immutable.
        return list(Snippet.lookup_single(tok.error, kind, snippet_mapping, tok.expect(Token.STRING)))
    else:
        tok.push_back(tok_typ, tok_value)

    val_list: list[T] = []
    tok.expect(Token.BRACK_OPEN)
    for token, first_value in tok.skipping_newlines():
        if token is Token.BRACK_CLOSE:
            return val_list
        elif token is Token.DIRECTIVE and first_value == 'snippet':
            # Include an existing list of values, maybe with inline values.
            key = tok.expect(Token.STRING)
            val_list.extend(Snippet.lookup_single(tok.error, kind, snippet_mapping, key))
            continue
        elif token is not Token.STRING:
            raise tok.error(token, first_value)

        vals = _read_colon_list(tok, had_colon=False)

        end_token, tok_value = tok()
        if end_token is Token.BRACK_OPEN:
            val_tags = read_tags(tok, Token.BRACK_CLOSE)
        else:
            val_tags = frozenset()
            tok.push_back(end_token, tok_value)
        val_list.append(parse(tok, error_desc, first_value, vals, val_tags))
    raise tok.error(Token.EOF)


def _parse_flags(
    tok: BaseTokenizer, error_desc: str,
    first_value: str, vals: list[str], tags: TagsSet,
) -> SpawnFlags:
    """Parse a line into a flags array member."""
    power: float
    if first_value.startswith(("^", "2^")):
        # Extension syntax - '2^4' == 16, '^0' == 1, etc.
        try:
            power = int(first_value.removeprefix('2').removeprefix('^'))
        except ValueError:
            raise tok.error(
                'SpawnFlags must be integer values or (2^X), not "{}" (in {})!',
                first_value,
                error_desc,
            ) from None
        spawnflag = 2**power
    else:
        try:
            spawnflag = int(first_value)
        except ValueError:
            raise tok.error(
                'SpawnFlags must be integer values or (2^X), not "{}" (in {})!',
                first_value,
                error_desc,
            ) from None
        try:
            power = math.log2(spawnflag)
        except ValueError:
            power = 0.5  # Force the following code to raise
        if power != round(power):
            raise tok.error(
                'SpawnFlags must be powers of two, not {} (in {})!',
                spawnflag,
                error_desc,
            )
    # Spawnflags can have a default, others may not.
    if len(vals) == 2:
        default = vals[1].strip() == '1'
    elif len(vals) == 1:
        default = True
    elif len(vals) == 0:
        raise tok.error('Expected value for spawnflags, got none!')
    else:
        raise tok.error(
            'Too many values for spawnflags definition in ({}):\n{}',
            error_desc, vals,
        )
    name = vals[0]
    # We optionally prepend [64] to spawnflags to show the numeric value.
    # Make sure we strip those to prevent duplication.
    generated_num = f'[{spawnflag}]'
    if name.startswith(generated_num):
        name = name[len(generated_num):].lstrip()
    return spawnflag, name, default, tags


def _parse_choices(
    tok: BaseTokenizer, error_desc: str,
    first_value: str, vals: list[str], tags: TagsSet,
) -> Choices:
    """Parse a line into a choices array member."""
    if len(vals) == 1:
        return (first_value, vals[0], tags)
    elif len(vals) == 0:
        raise tok.error('Expected value for choices, got none (in {})!', error_desc)
    else:
        raise tok.error(
            'Too many values for choices definition in ({}):\n{}',
            error_desc, vals,
        )


def read_tags(tok: BaseTokenizer, end_bracket: Token) -> TagsSet:
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
        elif token is end_bracket:
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
    error: Callable[[str], BaseException] = ValueError,
) -> TagsSet:
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


def match_tags(search: Container[str], tags: Collection[str]) -> bool:
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
        start = tag[:1]
        if start in ('!', '-'):
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


@attrs.frozen
class Snippet(Generic[ValueT_co]):
    """A part of some definition which has been given a name to be reused."""
    name: str
    source_path: str  # File it was defined in.
    source_line: int  # Line number.
    value: ValueT_co

    @classmethod
    def lookup_single(
        cls,
        error: Callable[[str], BaseException],
        kind: str,
        mapping: SnippetLookupDict[T],
        key: str,
    ) -> T:
        """Locate the one snippet using the specified mapping.

        If not found or having multiple, `error` is used to create the exception to raise, using `kind` as the snippet
        type.
        """
        try:
            snippets = mapping[key.casefold()]
        except KeyError:
            names = {snip.name for snip_list in mapping.values() for snip in snip_list}
            raise error(f'Snippet "{key}" does not exist. Known {kind} snippets: {sorted(names)}') from None
        try:
            [snip] = snippets
            return snip.value
        except ValueError:
            raise error(f'Multiple snippets with name "{key}" exist!') from None

    @classmethod
    def lookup_multi(
        cls,
        error: Callable[[str], BaseException],
        kind: str,
        mapping: SnippetLookupDict[T],
        key: str,
    ) -> Iterator[T]:
        """Locate all snippets with a key using the specified mapping.

        If not found, `error` is used to create the exception to raise, using `kind` as the snippet
        type.
        """
        try:
            snippets = mapping[key.casefold()]
        except KeyError:
            names = {snip.name for snip_list in mapping.values() for snip in snip_list}
            raise error(f'Snippet "{key}" does not exist. Known {kind} snippets: {sorted(names)}') from None
        else:
            for snip in snippets:
                yield snip.value

    @classmethod
    def _add(
        cls,
        kind: str, mapping: SnippetDict[T],
        path: str, line: int, name: str,
        value: T, replace: bool = False,
    ) -> None:
        """Create and add a snippet to the mapping, raising an error on collisions."""
        key = name.casefold()
        snip = Snippet(name, path, line, value)
        if replace:
            if key not in mapping:
                raise ValueError(
                    f'Tried to replace {kind} snippet with name "{name}", but none exist!'
                )
            mapping[key] = [snip]
        else:
            try:
                existing = mapping[key]
            except KeyError:
                mapping[key] = [snip]
            else:
                raise ValueError(
                    f'Two {kind} snippets were defined with the name "{name}":\n'
                    f'- {existing[0].source_path}:{existing[0].source_line}\n'
                    f'- {path}:{line}'
                )

    @classmethod
    def _append(
        cls, mapping: SnippetDict[T],
        path: str, line: int,
        name: str, value: T,
    ) -> None:
        """Create and add a snippet to the mapping, adding to existing values."""
        snip = Snippet(name, path, line, value)
        key = name.casefold()
        try:
            existing = mapping[key]
        except KeyError:
            mapping[key] = [snip]
        else:
            for exist in existing:
                if exist.value == value:
                    return
            existing.append(snip)

    # noinspection PyProtectedMember
    @classmethod
    def parse(cls, fgd: FGD, path: str, tokeniser: BaseTokenizer) -> None:
        """Parse snippet definitions in a FGD."""
        definition_line = tokeniser.line_num  # Before further parsing.
        snippet_kind = tokeniser.expect(Token.STRING).casefold()

        tok, tok_value = tokeniser()
        if tok is Token.STRING and tok_value == 'replace':
            replace = True
        else:
            replace = False
            tokeniser.push_back(tok, tok_value)

        snippet_id = tokeniser.expect(Token.STRING)
        tokeniser.expect(Token.EQUALS)
        error_desc = f'snippet "{path}:{snippet_id}'

        if snippet_kind in ('desc', 'description'):
            desc = tokeniser.expect(Token.STRING)
            while True:
                tok_type, tok_value = tokeniser()
                if tok_type is Token.PLUS:
                    desc += tokeniser.expect(Token.STRING)
                elif tok_type is Token.NEWLINE:
                    break
                else:
                    raise tokeniser.error(tok_type, tok_value)
            cls._add(
                'description', fgd.snippet_desc,
                path, definition_line, snippet_id,
                desc, replace=replace,
            )
        # These two can both use and produce snippets, producing a bit of redundancy.
        elif snippet_kind == 'choices':
            cls._add(
                'choices list', fgd.snippet_choices,
                path, definition_line, snippet_id,
                _parse_colon_array(
                    tokeniser, error_desc,
                    'choices list', fgd.snippet_choices, _parse_choices,
                ),  replace=replace,
            )
        elif snippet_kind in ('flags', 'spawnflags'):
            cls._add(
                'flags list', fgd.snippet_flags,
                path, definition_line, snippet_id,
                _parse_colon_array(
                    tokeniser, error_desc,
                    'flags list', fgd.snippet_flags, _parse_flags,
                ), replace=replace,
            )
        elif snippet_kind in ('kv', 'keyvalue'):
            cls._parse_kv_io(
                fgd, tokeniser,
                fgd.snippet_keyvalue, cls._parse_kv,
                path, definition_line, error_desc, replace,
                snippet_id,
            )
        elif snippet_kind == 'input':
            cls._parse_kv_io(
                fgd, tokeniser,
                fgd.snippet_input, cls._parse_io,
                path, definition_line, error_desc, replace,
                snippet_id,
            )
        elif snippet_kind == 'output':
            cls._parse_kv_io(
                fgd, tokeniser,
                fgd.snippet_output, cls._parse_io,
                path, definition_line, error_desc, replace,
                snippet_id,
            )
        else:
            raise tokeniser.error(
                f'Unknown snippet type "{snippet_kind}" for snippet "{path}:{snippet_id}"!'
            )

    @staticmethod
    def _parse_kv(fgd: FGD, tokeniser: BaseTokenizer, error_desc: str) -> tuple[TagsSet, KVDef]:
        """Callable for _parse_kv_io."""
        kv_name = tokeniser.expect(Token.STRING)
        # noinspection PyProtectedMember
        return KVDef._parse(fgd, kv_name, tokeniser, error_desc)

    @staticmethod
    def _parse_io(fgd: FGD, tokeniser: BaseTokenizer, _: str) -> tuple[TagsSet, IODef]:
        """Callable for _parse_kv_io."""
        # noinspection PyProtectedMember
        return IODef._parse(fgd, tokeniser)

    @classmethod
    def _parse_kv_io(
        cls,
        fgd: FGD, tokeniser: BaseTokenizer,
        field: SnippetDict[T],
        parse: Callable[[FGD, BaseTokenizer, str], T],
        path: str, definition_line: int,
        error_desc: str, replace: bool,
        snippet_id: str,
    ) -> None:
        """Parse a keyvalues, input or output definition."""
        tok, tok_value = tokeniser()
        if tok is Token.BRACK_OPEN:
            # We have multiple.
            while True:
                tok, tok_value = tokeniser()
                if tok is Token.NEWLINE:
                    continue
                elif tok is Token.BRACK_CLOSE:
                    break
                else:
                    tokeniser.push_back(tok, tok_value)
                    # We're multi-line, use the line for each definition
                    line = tokeniser.line_num
                    if replace:
                        field[snippet_id.casefold()] = [Snippet(
                            snippet_id, path, line,
                            parse(fgd, tokeniser, error_desc),
                        )]
                        replace = False
                    else:
                        cls._append(
                            field,
                            path, line, snippet_id,
                            parse(fgd, tokeniser, error_desc),
                        )
        else:
            # One value.
            tokeniser.push_back(tok, tok_value)
            if replace:
                field[snippet_id.casefold()] = [Snippet(
                    snippet_id, path, definition_line,
                    parse(fgd, tokeniser, error_desc),
                )]
            else:
                cls._append(
                    field,
                    path, definition_line, snippet_id,
                    parse(fgd, tokeniser, error_desc),
                )


@attrs.frozen
class Resource:
    """Resources used by an entity, with filetype.

    If the tags mapping is present, that indicates branch features that should/should not be
    present. Examples: 'episodic' (vs HL2), 'mapbase'.
    """
    filename: str
    type: FileType = FileType.GENERIC
    tags: TagsSet = frozenset()

    @classmethod
    def mdl(cls, filename: str, tags: TagsSet = frozenset(), skinset: Union[Sequence[int], int]=()) -> Resource:
        """Create a resource definition for a model.

        Models have an optional `filename#1,2,3` suffix to specify which skins are used.
        """
        if isinstance(skinset, int):
            filename = f'{filename}#{skinset}'
        elif skinset:
            filename = f'{filename}#{",".join(map(str, skinset))}'
        return cls(filename, FileType.MODEL, tags)

    @classmethod
    def mat(cls, filename: str, tags: TagsSet = frozenset()) -> Resource:
        """Create a resource definition for a material."""
        return cls(filename, FileType.MATERIAL, tags)

    @classmethod
    def snd(cls, filename: str, tags: TagsSet = frozenset()) -> Resource:
        """Create a resource definition for a soundscript."""
        return cls(filename, FileType.GAME_SOUND, tags)

    @classmethod
    def part(cls, filename: str, tags: TagsSet = frozenset()) -> Resource:
        """Create a resource definition for a particle system."""
        return cls(filename, FileType.PARTICLE_SYSTEM, tags)

    @classmethod
    def weapon_script(cls, filename: str, tags: TagsSet = frozenset()) -> Resource:
        """Create a resource definition for a weapon script."""
        return cls(filename, FileType.WEAPON_SCRIPT, tags)


@attrs.define(eq=True, repr=False, hash=False)
class Helper:
    """Base class for representing helper() commands in the header of an entity.

    These mainly add visual widgets in Hammer's views for manipulating and
    previewing keyvalues.

    This should not be instantiated, only subclasses in _fgd_helpers.
    """
    # The HelperType which this implements.
    TYPE: ClassVar[Optional[HelperTypes]] = None
    IS_EXTENSION: ClassVar[bool] = False  # true for our extensions to the format.

    tags: TagsSet = attrs.field(kw_only=True, default=frozenset())

    @classmethod
    def parse(cls, args: list[str], tags: TagsSet) -> Self:
        """Parse this helper from the given arguments.

        The default implementation expects no arguments.
        """
        if args:
            name = cls.TYPE.name if cls.TYPE is not None else cls.__name__
            raise ValueError(f'No arguments accepted by {name}()!')
        return cls(tags=tags)

    def export(self) -> list[str]:
        """Produce the argument text to recreate this helper type."""
        return []

    def get_resources(self, entity: EntityDef) -> Iterable[str]:
        """Return the resources used by this helper."""
        return ()

    def overrides(self) -> Collection[HelperTypes]:
        """Specify which types can be overriden by this.

        If any of these helper types are present before this type, they're
        redundant and can be removed.
        For example size() is ignored if a studio() is present after it.
        """
        return ()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(tags={self.tags!r})'


@attrs.define
class UnknownHelper(Helper):
    """Represents an unknown helper.

    This is intended only as a method for roundtripping FGDs with unexpected helpers.
    If you need to interact with an unknown helper, please file an issue/PR to get it implemented
    properly.
    """
    TYPE: ClassVar[Optional[HelperTypes]] = None
    name: str  #: The name of the helper.
    args: list[str]  #: The raw parameters passed to this helper. This is not stable.

    def export(self) -> list[str]:
        """Produce the argument text to recreate this helper type."""
        return self.args[:]

    __hash__ = None


HelperT = TypeVar('HelperT', bound=Helper)


@attrs.define(order=True, hash=True, eq=True)
class AutoVisgroup:
    """Represents one of the autovisgroup options that can be set.

    Due to how these are coded into Hammer, our representation is rather strange.
    We put all the groups into a single dictionary, and on each specify the name
    of the parent. Note they're case-sensitive, and can include punctuation.
    """
    name: str
    parent: str = attrs.field(hash=False, eq=False, order=False)
    ents: set[str] = attrs.field(factory=set, hash=False, eq=False, order=False)


class EntAttribute:
    """Common base class for IODef and KVDef."""
    name: str
    _type: Union[ValueTypes, str] = attrs.field(alias='type')
    desc: str

    def __init__(self) -> None:
        raise TypeError('EntAttribute is abstract, it cannot be instantiated!')

    @property
    def type(self) -> ValueTypes:
        """The type of this field.

        Custom types can be set as a string, but will appear as :py:attr:`ValueTypes.STRING`.
        To retrieve the custom name, access :py:attr:`custom_type`.
        """
        if isinstance(self._type, str):
            return ValueTypes.STRING
        else:
            return self._type

    @type.setter
    def type(self, value: Union[ValueTypes, str]) -> None:
        self._type = value

    @property
    def custom_type(self) -> Optional[str]:
        """If a custom type is specified, return the type name."""
        if isinstance(self._type, str):
            return self._type
        else:
            return None


@attrs.define
class KVDef(EntAttribute):
    """Represents a keyvalue that may be set on entities

    If the type is choices or spawnflags, ``val_list`` is required:
    * For choices it's a list of (value, name, tags) tuples.
    * For spawnflags it's a list of (bitflag, name, default, tags) tuples.
    """
    name: str
    _type: Union[ValueTypes, str] = attrs.field(alias='type')
    disp_name: str
    default: str = ''
    desc: str = ''
    val_list: Union[list[SpawnFlags], list[Choices], None] = None
    #: Causes Hammer to prevent this keyvalue from being changed.
    #: Doesn't force the value to the default though.
    readonly: bool = False
    #: If set, this value is shown in the Entity Report list.
    #: Usually applied to things like a sprite's material or a prop's model.
    reportable: bool = False
    #: Extension, marks keyvalues which are only used to preview things in Hammer.
    editor_only: bool = False

    @property
    def choices_list(self) -> list[Choices]:
        """Check that the keyvalues are CHOICES type, and then return val_list.

        This isolates the type ambiguity of the attr.
        """
        if self._type is not ValueTypes.CHOICES:
            raise TypeError
        if self.val_list is None:
            lst: list[tuple[str, str, TagsSet]] = []
            self.val_list = lst
        return cast('list[Choices]', self.val_list)

    @property
    def flags_list(self) -> list[SpawnFlags]:
        """Check that the keyvalues are SPAWNFLAGS type, and then return val_list.

        This isolates the type ambiguity of the attr.
        """
        if self._type is not ValueTypes.SPAWNFLAGS:
            raise TypeError
        if self.val_list is None:
            lst: list[SpawnFlags] = []
            self.val_list = lst
        return cast('list[SpawnFlags]', self.val_list)

    def copy(self) -> KVDef:
        """Create a duplicate of this keyvalue."""
        return KVDef(
            self.name,
            self._type,
            self.disp_name,
            self.default,
            self.desc,
            # Always copy this.
            self.val_list.copy() if self.val_list else None,
            self.readonly,
            self.reportable,
            self.editor_only,
        )

    __copy__ = copy

    def __deepcopy__(self, memodict: dict[int, Any]) -> KVDef:
        return KVDef(
            self.name,
            self._type,
            self.disp_name,
            self.default,
            self.desc,
            self.val_list.copy() if self.val_list else None,
            self.readonly,
            self.reportable,
            self.editor_only,
        )

    def known_options(self) -> Iterator[str]:
        """Use the default value and value list to determine values this can be set to."""
        if self._type is ValueTypes.CHOICES:
            options = {val_list[0] for val_list in self.choices_list}
            options.add(self.default)
            yield from options
        elif self._type is ValueTypes.SPAWNFLAGS:
            for bitflag, name, default, tags in self.flags_list:
                yield str(bitflag)
        else:
            yield self.default

    @classmethod
    def _parse(
        cls,
        fgd: FGD, name: str, tok: BaseTokenizer,
        error_desc: str,
        ignore_unknown_valuetype: bool = False,
    ) -> tuple[TagsSet, KVDef]:
        """Parse a keyvalue definition."""
        is_readonly = show_in_report = editor_only = had_colon = legacy_report = False
        # Next is either the value type parens, or a tags brackets.
        if tok.peek()[0] is Token.BRACK_OPEN:
            tok()  # Skip bracket.
            tags = read_tags(tok, Token.BRACK_CLOSE)
        else:
            tags = frozenset()

        tok.expect(Token.PAREN_OPEN)
        raw_value_type = tok.expect(Token.STRING)
        tok.expect(Token.PAREN_CLOSE)
        if raw_value_type.startswith('*'):
            # Old format for specifying 'reportable' flag.
            # Valve does allow both forms simultaneously.
            legacy_report = True
            raw_value_type = raw_value_type[1:]

        val_typ: Union[ValueTypes, str]
        try:
            val_typ = VALUE_TYPE_LOOKUP[raw_value_type.casefold()]
        except KeyError:
            msg = f'Unknown keyvalue type "{raw_value_type}"!'
            if ignore_unknown_valuetype:
                warnings.warn(msg, SyntaxWarning, stacklevel=2)
                val_typ = raw_value_type
            else:
                raise tok.error(msg) from None

        # Look for the 'readonly', 'report' and 'editor' flags.
        # Valve requires 'readonly report' in that order, but we'll allow any order.
        next_token, key_flag = tok()
        while next_token is Token.STRING:
            key_flag_fold = key_flag.casefold()
            if key_flag_fold == 'readonly':
                if is_readonly:
                    raise tok.error('Duplicate "readonly" flag!')
                is_readonly = True
            elif key_flag_fold == 'report':
                if show_in_report:
                    raise tok.error('Duplicate "report" flag!')
                show_in_report = True
            elif key_flag_fold == 'editor':
                if editor_only:
                    raise tok.error('Duplicate "editor" flag!')
                editor_only = True
            else:
                raise tok.error(next_token, key_flag)
            next_token, key_flag = tok()
        show_in_report |= legacy_report

        has_equal: Optional[Token] = None
        kv_vals: Optional[list[str]] = None
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
            raise tok.error(next_token, key_flag)
        if kv_vals is None:
            kv_vals = _read_colon_list(tok, had_colon, 2, fgd.snippet_desc)
            has_equal, _ = tok()
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
        # Read the choices in the [].
        val_list: Union[list[Choices], list[SpawnFlags], None]
        if isinstance(val_typ, ValueTypes) and val_typ.has_list:
            if has_equal is not Token.EQUALS:
                raise tok.error('No list provided for "{}" value type!', val_typ.name)
            if val_typ is ValueTypes.CHOICES:
                val_list = _parse_colon_array(
                    tok, error_desc,
                    'choices list', fgd.snippet_choices, _parse_choices,
                )
            elif val_typ is ValueTypes.SPAWNFLAGS:
                val_list = _parse_colon_array(
                    tok, error_desc,
                    'flags list', fgd.snippet_flags, _parse_flags,
                )
            else:  # No others have a list.
                raise AssertionError(val_typ)
        else:
            val_list = None
            if has_equal is Token.EQUALS:
                raise tok.error(
                    '"{}" value types can\'t have lists!',
                    getattr(val_typ, "name", val_typ),
                )

        return tags, KVDef(
            name=name,
            type=val_typ,
            desc=kv_desc,
            disp_name=disp_name,
            default=default,
            val_list=val_list,
            readonly=is_readonly,
            reportable=show_in_report,
            editor_only=editor_only,
        )

    def export(
        self,
        file: FileWText,
        tags: Collection[str] = (),
        label_spawnflags: bool = True,
        custom_syntax: bool = True,
        old_report: bool = False,
    ) -> None:
        """Write this back out to a FGD file."""
        file.write('\t' + self.name)
        if tags and custom_syntax:
            file.write(f'[{", ".join(sorted(tags))}]')
        file.write('(*' if self.reportable and old_report else '(')
        if isinstance(self._type, ValueTypes):
            file.write(f'{self._type.value})')
        else:
            file.write(f'{self._type})')

        if self.editor_only and custom_syntax:
            file.write(' editor')
        # Note Valve requires this order if both are present.
        if self.readonly:
            file.write(' readonly')
        if self.reportable and not old_report:
            file.write(' report')

        if self._type is not ValueTypes.SPAWNFLAGS:
            # Spawnflags never use names!
            file.write(': ')
            _write_longstring(file, custom_syntax, self.disp_name, indent='\t')

        default = self.default
        if not default and self.type is ValueTypes.BOOL:
            # This has to be present.
            default = '0'

        if default:
            default_str = str(default)
            # We can write unquoted integers, but nothing else.
            if all(x in '0123456789-' for x in default_str):
                file.write(' : ' + default_str)
            else:
                file.write(f' : "{default_str}"')
            if self.desc:
                file.write(' : ')
        else:
            if self.desc:
                file.write(' : : ')

        if self.desc:
            _write_longstring(file, custom_syntax, self.desc, indent='\t')

        if isinstance(self._type, ValueTypes) and self._type.has_list:
            file.write(' =\n\t\t[\n')
            if self._type is ValueTypes.SPAWNFLAGS:
                # Empty tuple handles a None value.
                for index, name, flag_default, tags in self.flags_list:
                    file.write(f'\t\t{index}: ')
                    # Newlines aren't functional here, just replace.
                    name = name.replace('\n', ' ')
                    _write_longstring(
                        file,
                        custom_syntax,
                        f'[{index}] {name}' if label_spawnflags else name,
                        indent='\t\t',
                    )
                    file.write(' : 1' if flag_default else ' : 0')
                    if tags and custom_syntax:
                        file.write(f' [{", ".join(sorted(tags))}]\n')
                    else:
                        file.write('\n')
            elif self._type is ValueTypes.CHOICES:
                for value, name, tags in self.choices_list:
                    # Numbers can be unquoted, everything else cannot.
                    try:
                        float(value)
                    except ValueError:
                        value = f'"{value}"'

                    file.write(f'\t\t{value}: ')
                    # Newlines aren't functional here, just replace.
                    _write_longstring(file, False, name.replace('\n', ' '), indent='\t\t')
                    if tags and custom_syntax:
                        file.write(f' [{", ".join(sorted(tags))}]\n')
                    else:
                        file.write('\n')
            else:
                raise AssertionError('No other has-list types possible!')
            file.write('\t\t]')  # No newline, done unconditionally below.

        file.write('\n')


@attrs.define
class IODef(EntAttribute):
    """Represents an input or output for an entity."""
    name: str
    _type: Union[ValueTypes, str] = attrs.field(alias='type', default=ValueTypes.VOID)  # Most IO has no parameter.
    desc: str = ''

    def copy(self) -> IODef:
        """Create a duplicate of this IODef."""
        return IODef(self.name, self._type, self.desc)

    __copy__ = copy

    def __deepcopy__(self, memodict: dict[int, Any]) -> IODef:
        return IODef(self.name, self._type, self.desc)

    @classmethod
    def _parse(
        cls,
        fgd: FGD,
        tok: BaseTokenizer,
        ignore_unknown_valuetype: bool = False,
    ) -> tuple[TagsSet, IODef]:
        """Parse I/O definitions in an entity."""
        name = tok.expect(Token.STRING)

        # Next is either the value type parens, or a tags brackets.
        if tok.peek()[0] is Token.BRACK_OPEN:
            tok()
            tags = read_tags(tok, Token.BRACK_CLOSE)
        else:
            tags = frozenset()
        tok.expect(Token.PAREN_OPEN)
        raw_value_type = tok.expect(Token.STRING)
        tok.expect(Token.PAREN_CLOSE)

        val_typ: Union[ValueTypes, str]
        if raw_value_type == 'ehandle':
            # This is a duplicate (deprecated) name, but only for I/O.
            val_typ = ValueTypes.EHANDLE
        else:
            try:
                val_typ = VALUE_TYPE_LOOKUP[raw_value_type.casefold()]
            except KeyError:
                msg = f'Unknown keyvalue type "{raw_value_type}"!'
                if ignore_unknown_valuetype:
                    warnings.warn(msg, SyntaxWarning, stacklevel=2)
                    val_typ = raw_value_type
                else:
                    raise tok.error(msg) from None

        # Read desc
        io_vals = _read_colon_list(tok, False, 0, fgd.snippet_desc)

        tok.expect(Token.NEWLINE)

        if io_vals:
            try:
                [io_desc] = io_vals
            except ValueError:
                raise tok.error('Too many values for IO definition!') from None
        else:
            io_desc = ''

        return tags, cls(name, val_typ, io_desc)

    def export(
        self,
        file: FileWText,
        io_type: str,
        tags: Collection[str] = (),
        custom_syntax: bool = True,
    ) -> None:
        """Write this back out to a FGD file.

        io_type should be "input" or "output".
        """
        file.write(f'\t{io_type} {self.name}')

        if custom_syntax and tags:
            file.write(f'[{", ".join(sorted(tags))}]')

        # Special case, bool is "boolean" on values, "bool" on IO...
        if self._type is ValueTypes.BOOL:
            file.write('(bool)')
        elif isinstance(self._type, ValueTypes):
            file.write(f'({VALUE_TO_IO_DECAY[self._type].value})')
        else:
            file.write(f'({self._type})')

        if self.desc:
            file.write(' : ')
            _write_longstring(file, custom_syntax, self.desc, indent='\t')
        file.write('\n')


class _EntityView(Generic[T]):
    """Provides a view over entity keyvalues, inputs, or outputs."""
    __slots__ = ['_ent', '_attr', '_disp_attr']

    # Note, we expect the maps to have casefolded their keys.
    def __init__(self, ent: EntityDef, attr_name: str, disp_name: str) -> None:
        self._ent = ent
        self._attr = attr_name
        self._disp_attr = disp_name

    def __repr__(self) -> str:
        return f'{self._ent!r}.{self._disp_attr}'

    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        """We're private, so we should be the only instance for a given Entity."""
        return other is self

    def _maps(
        self,
        ent: Optional[EntityDef] = None,
    ) -> Iterator[Mapping[str, Mapping[TagsSet, T]]]:
        """Yield all the mappings which we need to look through."""
        if ent is None:
            ent = self._ent

        yield getattr(ent, self._attr)
        for base in ent.bases:
            if isinstance(base, EntityDef):
                yield from self._maps(base)

    def __getitem__(self, name: Union[str, tuple[str, Collection[str]]]) -> T:
        """Lookup the value in the entity.

        Either obj['name'], or obj['name', {tags}] is accepted.
        """
        if isinstance(name, str):
            search_tags = None
        elif isinstance(name, tuple):
            name, tags_iter = name
            if isinstance(tags_iter, str):
                search_tags = frozenset([tags_iter.upper()])
            else:
                search_tags = frozenset({t.upper() for t in tags_iter})
        else:
            raise TypeError(f'Expected str or (str, Collection[str]), got {name!r}')
        name = name.casefold()
        for ent_map in self._maps():
            try:
                tag_map = ent_map[name]
            except KeyError:
                continue
            # Ignore tags if not provided.
            if search_tags is None:
                return tag_map[frozenset()]

            # Force longer more-specific tags to match first.
            for tags, value in sorted(
                tag_map.items(),
                key=lambda t: len(t[0]),
                reverse=True,
            ):
                if match_tags(search_tags, tags):
                    return value
        raise KeyError((name, search_tags))

    def __setitem__(self, name: Union[str, tuple[str, Collection[str]]], value: T) -> None:
        """Set the value in the entity. This never affects bases.

        This only overwrites values with the exact same tag."""
        if isinstance(name, str):
            tags: TagsSet = frozenset()
        elif isinstance(name, tuple):
            name, tags_iter = name
            if isinstance(tags_iter, str):
                tags = frozenset([tags_iter.upper()])
            else:
                tags = frozenset({t.upper() for t in tags_iter})
        else:
            raise TypeError(f'Expected str or (str, Collection[str]), got {name!r}')
        ent_map: dict[str, dict[TagsSet, T]] = getattr(self._ent, self._attr)
        name = name.casefold()
        try:
            tag_map = ent_map[name]
        except KeyError:
            ent_map[name] = {tags: value}
        else:
            tag_map[tags] = value

    def __delitem__(self, name: Union[str, tuple[str, Collection[str]]]) -> None:
        """Remove the value from the entity. This never affects bases.

        If a single string is provided, all tag versions are removed."""
        ent_map: dict[str, dict[TagsSet, T]] = getattr(self._ent, self._attr)
        if isinstance(name, str):
            del ent_map[name.casefold()]  # Or keyError
            return
        elif not isinstance(name, tuple):
            raise TypeError(f'Expected str or (str, Collection[str]), got {name!r}')

        name, tags_iter = name
        if isinstance(tags_iter, str):
            tags = frozenset([tags_iter.upper()])
        else:
            tags = frozenset({t.upper() for t in tags_iter})
        try:
            del ent_map[name.casefold()][tags]
        except KeyError:
            raise KeyError((name, tags)) from None

    def __iter__(self) -> Iterator[str]:
        """Yields all keys this object has."""
        seen: set[str] = set()
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
        seen: set[str] = set()
        for ent_map in self._maps():
            seen.update(ent_map)
        return len(seen)


@attrs.define(slots=False, eq=False, repr=False)
class EntityDef:
    """A definition for an entity."""
    type: EntityTypes  #: The kind of entity.
    classname: str = ''  #: The classname of this entity, as originally typed.

    # These are (name) -> {tags: value} dicts.
    keyvalues: dict[str, dict[TagsSet, KVDef]] = attrs.field(kw_only=True, factory=dict)
    inputs: dict[str, dict[TagsSet, IODef]] = attrs.field(kw_only=True, factory=dict)
    outputs: dict[str, dict[TagsSet, IODef]] = attrs.field(kw_only=True, factory=dict)

    #: Keyvalues have an order. If not present in here, they appear at the end.
    kv_order: list[str] = attrs.field(kw_only=True, factory=list)

    #: The parent entity classes defined using ``base()`` helpers.
    bases: list[Union[EntityDef, str]] = attrs.field(kw_only=True, factory=list)
    #: All other helpers defined on the entity.
    helpers: list[Helper] = attrs.field(kw_only=True, factory=list)
    desc: str = attrs.field(default='', kw_only=True)

    # Views for accessing data among all the entities.
    kv: _EntityView[KVDef] = attrs.field(init=False)
    inp: _EntityView[IODef] = attrs.field(init=False)
    out: _EntityView[IODef] = attrs.field(init=False)

    #: A list of resources this entity may require. Use :py:func:`get_resources()` to recursively
    #: fetch sub-entity resources.
    # This is set to an empty tuple to represent an entity which has
    #: no `@resources` definition, as opposed to an empty list for an explicit empty resources
    #: definition. Use :py:func:`resources_defined()` to distinguish between these cases.
    resources: Sequence[Resource] = attrs.field(kw_only=True, default=())
    #: If set, the ``aliasof()`` helper was used. This entity should have 1 base, which this is
    #: simply an alternate classname for.
    is_alias: bool = attrs.field(kw_only=True, default=False)

    def __attrs_post_init__(self) -> None:
        """Setup Entity views."""
        self.kv = _EntityView(self, 'keyvalues', 'kv')
        self.inp = _EntityView(self, 'inputs', 'inp')
        self.out = _EntityView(self, 'outputs', 'out')

    @classmethod
    def parse(
        cls,
        fgd: FGD,
        tok: Tokenizer,
        ent_type: EntityTypes,
        eval_bases: bool = True,
        eval_extensions: bool = True,
        ignore_unknown_valuetype: bool = False
    ) -> None:
        """Parse an entity definition from an FGD file.

        The ``@PointClass`` etc keyword should already have been read, and is passed as ``ent_type``.
        """

        entity = cls(ent_type)

        # First parse the bases part - lots of name(args) sections until an '='.
        ext_autovisgroups: list[list[str]] = []
        help_type: Optional[HelperTypes] = None
        help_type_cust: Optional[str] = None
        help_tags: Optional[TagsSet] = None
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
                        entity.helpers.append(HELPER_IMPL[help_type].parse(
                            [], help_tags or frozenset(),
                        ))
                    except ValueError as exc:
                        raise tok.error(
                            'Invalid helper arguments for {}{}()',
                            help_type.value,
                            f'[{" ".join(help_tags)}]' if help_tags else '',
                        ) from exc
                    help_type = help_tags = None
                    # Then repeat this token so it's parsed.
                    tok.push_back(token, token_value)
                continue
            elif token is Token.BRACK_OPEN:
                if help_type is None and help_type_cust is None:
                    raise tok.error('Tags specified without helper type')
                if help_tags is not None:
                    raise tok.error('Duplicate helper tags')
                help_tags = read_tags(tok, Token.BRACK_CLOSE)

            elif token is Token.PAREN_OPEN:
                if help_type is None and help_type_cust is None:
                    raise tok.error('Args without helper type! ({!r})', token_value)

                args = []

                if help_type is HelperTypes.CUBE:
                    # This is parsed directly in gdclass.cpp, and has a meaningful comma.
                    args += [
                        tok.expect(Token.STRING),
                        tok.expect(Token.STRING),
                        tok.expect(Token.STRING)
                    ]
                    if tok.peek()[0] is Token.COMMA:
                        tok.expect(Token.COMMA)
                        args += [
                            tok.expect(Token.STRING),
                            tok.expect(Token.STRING),
                            tok.expect(Token.STRING)
                        ]
                    tok.expect(Token.PAREN_CLOSE)
                elif help_type is HelperTypes.EXT_AUTO_VISGROUP:
                    # Parse specially to handle commas.
                    partial_vis: list[str] = []
                    for arg_tok, arg_val in tok:
                        if arg_tok is Token.PAREN_CLOSE:
                            if partial_vis:
                                args.append(' '.join(partial_vis))
                            break
                        elif arg_tok is Token.STRING:
                            partial_vis.append(arg_val)
                        elif arg_tok is Token.COMMA:
                            if partial_vis:
                                args.append(' '.join(partial_vis))
                            partial_vis.clear()
                        else:
                            raise tok.error(arg_tok, arg_val)
                else:
                    # We need to allow + in args.
                    try:
                        tok.plus_operator = False
                        for arg_tok, arg_val in tok:
                            if arg_tok is Token.PAREN_CLOSE:
                                break
                            elif arg_tok is Token.STRING:
                                args.append(arg_val)
                            # Comma is actually irrelevant for all normal types other than size().
                            elif arg_tok is not Token.COMMA:
                                raise tok.error(arg_tok, arg_val)
                    finally:
                        tok.plus_operator = True

                if help_type_cust == 'aliasof':
                    # Extension, indicate that it's an alias. The args are then treated like base()
                    help_type_cust = None
                    help_type = HelperTypes.INHERIT
                    entity.is_alias = True

                if help_type_cust is not None:
                    entity.helpers.append(UnknownHelper(help_type_cust, args, tags=help_tags or frozenset()))
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
                        entity.helpers.append(HELPER_IMPL[help_type].parse(
                            args, help_tags or frozenset(),
                        ))
                    except (TypeError, ValueError) as exc:
                        raise tok.error(
                            'Invalid helper arguments for {}{}({}):\n',
                            help_type.value,
                            f'[{" ".join(help_tags)}]' if help_tags else '',
                            args,
                        ) from exc

                help_type = help_type_cust = help_tags = None

            elif token is Token.EQUALS:
                break
            else:
                raise tok.error(token)
        else:
            raise tok.error('Entity header never ended!')

        # We were waiting for arguments for the previous helper.
        # We need to add with none.
        if help_type_cust is not None:
            entity.helpers.append(UnknownHelper(help_type_cust, [], tags=help_tags or frozenset()))
        elif help_type:
            if help_type is HelperTypes.EXT_AUTO_VISGROUP or help_type is HelperTypes.INHERIT:
                raise tok.error('{}() requires at least one argument!', help_type.value)
            try:
                entity.helpers.append(HELPER_IMPL[help_type].parse([], help_tags or frozenset()))
            except ValueError as exc:
                raise tok.error(
                    'Invalid helper arguments for {}{}()',
                    help_type.value,
                    f'[{" ".join(help_tags)}]' if help_tags else '',
                ) from exc

        entity.classname = tok.expect(Token.STRING).strip()

        # We next might have a ':' then docstring before the [,
        # or directly to [.
        desc: Optional[list[str]] = None
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
                    raise tok.error(doc_token, token_value)
                desc.append(token_value)
            elif doc_token is Token.DIRECTIVE and token_value == 'snippet':
                if desc is None or desc:
                    # No colon yet, or we have text without '+' between
                    raise tok.error(doc_token, token_value)
                # Included from an earlier snippet.
                desc.extend(Snippet.lookup_single(
                    tok.error,
                    'description', fgd.snippet_desc,
                    tok.expect(Token.STRING),
                ))
            elif doc_token is Token.PLUS:
                if not desc:
                    raise tok.error('+ without string before it!')
                tok_typ, tok_val = tok()
                while tok_typ is Token.NEWLINE:
                    tok_typ, tok_val = tok()
                if tok_typ is Token.STRING:
                    desc.append(tok_val)
                elif tok_typ is Token.DIRECTIVE and tok_val == 'snippet':
                    desc.extend(Snippet.lookup_single(
                        tok.error, 'description', fgd.snippet_desc,
                        tok.expect(Token.STRING),
                    ))
                else:
                    raise tok.error(tok_typ, tok_val)
            elif doc_token is Token.BRACK_OPEN:
                if desc:
                    entity.desc = ''.join(desc)
                break
            else:
                raise tok.error(doc_token, token_value)

        # Now apply EXT_AUTO_VISGROUP, since we have the classname.
        for auto_visgroup in ext_autovisgroups:
            for vis_parent, vis_name in zip(auto_visgroup, auto_visgroup[1:]):
                try:
                    visgroup = fgd.auto_visgroups[vis_name.casefold()]
                except KeyError:
                    visgroup = fgd.auto_visgroups[vis_name.casefold()] = AutoVisgroup(vis_name, vis_parent)
                visgroup.ents.add(entity.classname)

        # Now parse keyvalues, and input/outputs
        while True:
            token, token_value = tok()
            if token is Token.BRACK_CLOSE:
                break  # End of this entity.

            if token is Token.NEWLINE:
                continue

            if token is Token.DIRECTIVE and token_value == 'snippet':
                value_kind = tok.expect(Token.STRING).casefold()
                key = tok.expect(Token.STRING)
                if value_kind == 'input':
                    for tags, io_def in Snippet.lookup_multi(tok.error, 'input', fgd.snippet_input, key):
                        io_tags_map = entity.inputs.setdefault(io_def.name.casefold(), {})
                        io_tags_map[tags] = io_def.copy()
                elif value_kind == 'output':
                    for tags, io_def in Snippet.lookup_multi(tok.error, 'output', fgd.snippet_output, key):
                        io_tags_map = entity.outputs.setdefault(io_def.name.casefold(), {})
                        io_tags_map[tags] = io_def.copy()
                elif value_kind == 'keyvalue':
                    for tags, kv_def in Snippet.lookup_multi(tok.error, 'keyvalue', fgd.snippet_keyvalue, key):
                        kv_tags_map = entity.keyvalues.setdefault(kv_def.name.casefold(), {})
                        if not kv_tags_map:
                            # New, add to the ordering.
                            entity.kv_order.append(kv_def.name.casefold())
                        kv_tags_map[tags] = kv_def.copy()
                else:
                    raise tok.error(
                        'Unknown snippet type "{}". Valid in this context: '
                        'input, output, keyvalue',
                        value_kind,
                    )
                continue

            # IO - keyword at the start.
            if token is not Token.STRING:
                raise tok.error(token, token_value)

            io_type = token_value.casefold()
            if io_type == 'input':
                # noinspection PyProtectedMember
                tags, io_def = IODef._parse(fgd, tok, ignore_unknown_valuetype)
                io_tags_map = entity.inputs.setdefault(io_def.name.casefold(), {})
                io_tags_map[tags] = io_def
            elif io_type == 'output':
                # noinspection PyProtectedMember
                tags, io_def = IODef._parse(fgd, tok, ignore_unknown_valuetype)
                io_tags_map = entity.outputs.setdefault(io_def.name.casefold(), {})
                io_tags_map[tags] = io_def
            elif io_type == '@resources':  # @resource block, format extension
                tok.expect(Token.BRACK_OPEN, skip_newline=True)
                # Append to existing, in case there's multiple blocks.
                resources: list[Resource] = list(entity.resources)
                for res_tok, res_tok_val in tok:
                    if res_tok is Token.STRING:
                        try:
                            res_type = RESTYPE_BY_NAME[res_tok_val.casefold()]
                        except KeyError:
                            raise tok.error('Unknown resource type "{}"!', res_tok_val) from None
                        filename = tok.expect(Token.STRING)
                        tags = frozenset()
                        token, tok_val = tok()
                        if token is Token.BRACK_OPEN:
                            tags = read_tags(tok, Token.BRACK_CLOSE)
                        resources.append(Resource(filename, res_type, tags))
                    elif res_tok is Token.BRACK_CLOSE:
                        break
                # Subtle: don't convert to tuple, we use () to represent unset resources.
                entity.resources = resources
            else:
                # noinspection PyProtectedMember
                tags, kv_def = KVDef._parse(fgd, token_value, tok, entity.classname, ignore_unknown_valuetype)
                kv_tags_map = entity.keyvalues.setdefault(kv_def.name.casefold(), {})
                if not kv_tags_map:
                    # New, add to the ordering.
                    entity.kv_order.append(kv_def.name.casefold())
                kv_tags_map[tags] = kv_def

        if eval_extensions and ent_type == EntityTypes.EXTEND:
            # Check for if the entry already exists. If it does, extend it.
            # Otherwise, we'll just store it in there...
            try:
                original_ent = fgd.entities[entity.classname.casefold()]
                original_ent.extend(entity)
            except KeyError:
                fgd.entities[entity.classname.casefold()] = entity
        else:
            fgd.entities[entity.classname.casefold()] = entity

    @classmethod
    def engine_def(cls, classname: str) -> EntityDef:
        """Return the specified entity from an internal copy of the Hammer Addons database.

        This can be used to identify the kind of keys/inputs/outputs present on an entity, as well
        as resources the entity requires/:external:cpp:func:`!Precache()`\\ es.

        :raises KeyError: If the classname is not found in the database.
        """
        databases = _load_engine_db()
        for dbase in databases:
            try:
                return deepcopy(dbase.get_ent(classname))
            except KeyError:
                pass
        
        raise KeyError(classname)

    @classmethod
    def engine_classes(cls) -> AbstractSet[str]:
        """Return a set of known entity classnames, from the Hammer Addons database."""
        databases = _load_engine_db()

        # If only one database exists, just pass on its set directly.
        if len(databases) == 1:
            return databases[0].get_classnames()

        classnames: set[str] = set()
        for dbase in databases:
            classnames |= dbase.get_classnames()

        return frozenset(classnames)

    def resources_defined(self) -> bool:
        """Check if any resources were defined for this entity, even if blank."""
        return self.resources != ()

    def __repr__(self) -> str:
        if self.type is EntityTypes.BASE:
            return f'<Entity Base "{self.classname}">'
        else:
            return f'<Entity {self.classname}>'

    def __deepcopy__(self, memodict: dict[int, Any]) -> EntityDef:
        """Handle copying ourselves, to eliminate lookups when not required."""
        copy = EntityDef.__new__(EntityDef)
        copy.type = self.type
        copy.classname = self.classname
        copy.kv_order = self.kv_order.copy()
        copy.bases = deepcopy(self.bases, memodict)
        copy.helpers = deepcopy(self.helpers, memodict)
        copy.desc = self.desc
        copy.resources = self.resources
        copy.is_alias = self.is_alias

        # Avoid copy for these, we know the tags-map is immutable.
        for val_key in ['keyvalues', 'inputs', 'outputs']:
            coll: dict[str, dict[TagsSet, Union[KVDef, IODef]]] = {}
            setattr(copy, val_key, coll)
            tags_map: dict[TagsSet, Union[KVDef, IODef]]
            for key, tags_map in getattr(self, val_key).items():
                coll[key] = {
                    key: value.copy()
                    for key, value in tags_map.items()
                }
        copy.kv = _EntityView(copy, 'keyvalues', 'kv')
        copy.inp = _EntityView(copy, 'inputs', 'inp')
        copy.out = _EntityView(copy, 'outputs', 'out')
        return copy

    def __getstate__(self) -> tuple[object, ...]:
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
            self.desc,
            self.resources,
            self.is_alias,
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
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
            self.desc,
            *resources,
        ) = state
        self.kv = _EntityView(self, 'keyvalues', 'kv')
        self.inp = _EntityView(self, 'inputs', 'inp')
        self.out = _EntityView(self, 'outputs', 'out')
        if resources:  # Backwards compat.
            [self.resources, self.is_alias] = resources

    @overload
    def get_helpers(self, typ: builtins.type[HelperT], tags: TagsSet = ...) -> Iterator[HelperT]: ...
    @overload
    def get_helpers(self, typ: str, tags: TagsSet = ...) -> Iterator[UnknownHelper]: ...

    def get_helpers(self, typ: Union[builtins.type[HelperT], str], tags: TagsSet = frozenset()) -> Iterator[Helper]:
        """Find all helpers with this specific type and tags."""
        if isinstance(typ, str):
            for helper in self.helpers:
                if isinstance(helper, UnknownHelper) and helper.name == typ and match_tags(tags, helper.tags):
                    yield helper
        else:
            for helper in self.helpers:
                if helper.TYPE == typ.TYPE and match_tags(tags, helper.tags):
                    yield helper

    def get_resources(
        self,
        ctx: ResourceCtx,
        *,
        ent: Optional[Entity],
        on_error: Callable[[str], object] = lambda err: None,
    ) -> Iterator[tuple[FileType, str]]:
        """Recursively fetch all the resources this entity may use, simulating ``Precache()``.

        :param ent: A specific entity to evaluate against. If not provided, functions will
            silently be skipped.
        :param ctx: Common information about the current map and game configuration. This is
            passed along to defined entclass functions, and is seperate, so it can be reused
            for many calls to this function.
        :param on_error: If provided, when functions or entities are missing this will be called
            with the specific error, and raised if it returns an exception type. If not
            set, lookups are ignored. Most exceptions can be passed directly here to cause that to
            be raised.
        """
        if not self.resources:
            # Nothing to do, skip making the sets/lists below.
            return
        # We can recurse, use two lists to avoid actual recursive calls.
        # Also track the checked classes, so we don't repeat ourselves.
        classes_checked = {self.classname}
        entities_checked: set[frozenset[tuple[str, str]]] = set()
        todo_ents: list[tuple[EntityDef, Optional[Entity]]] = [(self, ent)]
        todo_res: list[Resource] = []
        while todo_ents:
            (ent_def, ent) = todo_ents.pop()
            todo_res.extend(ent_def.resources)
            while todo_res:
                res = todo_res.pop()
                # Skip resources with bad tags, and also skip those with totally empty filenames.
                # The latter is usually just an unset keyvalue, not important.
                if not match_tags(ctx.tags, res.tags) or not res.filename:
                    continue
                if res.type is FileType.ENTITY:
                    if res.filename not in classes_checked:
                        try:
                            sub_ent = ctx.get_entdef(res.filename)
                        except LookupError:  # KeyError or IndexError
                            err = on_error(f'Missing entity definition: "{res.filename}"')
                            if isinstance(err, BaseException):
                                raise err from None
                            continue
                        classes_checked.add(res.filename)
                        # For entity recursions, we pass the same ent down.
                        todo_ents.append((sub_ent, ent))
                elif res.type is FileType.ENTCLASS_FUNC:
                    if ent is None:
                        ent = _fake_vmf.create_ent(ent_def.classname)
                    ent_key = frozenset({
                        (key, value) for key, value in
                        ent.items()
                        # Treat entities at different locations as the same.
                        # Ignore IDs and names (almost always unique)
                        if key not in {
                            'origin', 'angles',
                            'targetname', 'id', 'hammerid', 'nodeid',
                        }
                    })
                    if ent_key in entities_checked:
                        continue
                    entities_checked.add(ent_key)
                    try:
                        # noinspection PyProtectedMember
                        func = ctx._functions[res.filename]
                    except LookupError:
                        err = on_error(f'Missing function: "{res.filename}" in "{ent_def.classname}"')
                        if isinstance(err, BaseException):
                            raise err from None
                        continue
                    for sub_res in func(ctx, ent):
                        if isinstance(sub_res, Entity):
                            try:
                                sub_ent = ctx.get_entdef(sub_res['classname'])
                            except LookupError:  # KeyError or IndexError
                                err = on_error(f'Missing entity definition: "{sub_res["classname"]}"')
                                if isinstance(err, BaseException):
                                    raise err from None
                                continue
                            todo_ents.append((sub_ent, sub_res))
                            continue
                        else:
                            todo_res.append(sub_res)
                else:
                    yield (res.type, res.filename)

    def _iter_attrs(self) -> Iterator[dict[str, dict[TagsSet, EntAttribute]]]:
        """Iterate over both the keyvalues and I/O dicts.

        This is used when we want to deal with both in the same way.
        """
        return iter([self.keyvalues, self.inputs, self.outputs])  # type: ignore

    def strip_tags(self, tags: TagsSet) -> None:
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
                        if isinstance(value, KVDef) and value.val_list:
                            # Filter the value list as well, then discard tags.
                            # Use slicing/negative index to preserve the same number of args
                            value.val_list = [  # pyright: ignore
                                (*val, frozenset())
                                for (*val, tag) in value.val_list
                                if match_tags(tags, tag)
                            ]
                        break
                else:
                    del category[key]

    def export(
        self,
        file: FileWText,
        label_spawnflags: bool = True,
        custom_syntax: bool = True,
        old_report: bool = False,
    ) -> None:
        """Write the entity out to a FGD file.

        See :py:meth:`FGD.export()` for the meaning of the parameters.
        """
        # Make it look pretty: BaseClass
        file.write(f'@{self.type.value.title().replace("class", "Class")} ')
        if self.bases:
            file.write('base(')
            file.write(', '.join([
                (base.classname if isinstance(base, EntityDef) else base)
                for base in self.bases
            ]))
            file.write(') ')

        kv_order_list: list[str] = []

        for helper in self.helpers:
            args = helper.export()
            help_tags = f'[{", ".join(sorted(helper.tags))}]' if custom_syntax and helper.tags else ''
            if isinstance(helper, HelperExtOrderBy):
                # Even if custom syntax is off, still apply this helper.
                kv_order_list += [arg.casefold() for arg in args]
            if helper.IS_EXTENSION and not custom_syntax:
                continue
            if isinstance(helper, HelperHalfGridSnap):
                # Special case, no args.
                file.write(f'\n\thalfgridsnap{help_tags}')
            elif isinstance(helper, UnknownHelper):
                file.write(f'\n\t{helper.name}{help_tags}({", ".join(args)})')
            elif helper.TYPE is not None:
                file.write(f'\n\t{helper.TYPE.value}{help_tags}({", ".join(args)})')
            else:
                raise TypeError(f'Helper {helper!r} has no TYPE attr?')

        if self.helpers:
            file.write('\n')  # Put the classname on the following line.
        file.write(f'= {self.classname}')

        if self.desc:
            file.write(': ')
            _write_longstring(file, custom_syntax, self.desc, indent='\t\t')

        file.write('\n\t[\n')

        kv_order = {
            name: ind
            for ind, name in
            enumerate(kv_order_list or self.kv_order)
        }
        def kv_order_key(tup: tuple[str, dict[frozenset[str], KVDef]]) -> float:
            """Sort by position in kv_order. If not present add to the end."""
            return kv_order.get(tup[0], math.inf)

        for name, kv_map in sorted(self.keyvalues.items(), key=kv_order_key):
            for tags, kv in kv_map.items():
                kv.export(file, tags, label_spawnflags, custom_syntax, old_report)

        if self.inputs:
            file.write('\n\t// Inputs\n')

            for inp_map in self.inputs.values():
                for tags, inp in inp_map.items():
                    inp.export(file, 'input', tags, custom_syntax)

        if self.outputs:
            file.write('\n\t// Outputs\n')

            for out_map in self.outputs.values():
                for tags, out in out_map.items():
                    out.export(file, 'output', tags, custom_syntax)

        if custom_syntax and self.resources != ():
            file.write('\n\t@resources\n\t\t[\n')
            for res in self.resources:
                file.write(f'\t\t{RESTYPE_TO_NAME[res.type]} "{escape_text(res.filename)}"')
                if res.tags:
                    file.write(f' [{", ".join(sorted(res.tags))}]\n')
                else:
                    file.write('\n')
            file.write('\t\t]\n')

        file.write('\t]\n')

    def iter_bases(self) -> Iterator[EntityDef]:
        """Yield all base entities for this one.

        If an entity is repeated, it will only be yielded once.
        """
        return self._iter_bases({self})

    def _iter_bases(self, done: set[EntityDef]) -> Iterator[EntityDef]:
        for ent in self.bases:
            if ent in done or isinstance(ent, str):
                continue

            done.add(ent)
            yield ent
            yield from ent._iter_bases(done)

    def extend(self, other: EntityDef) -> bool:
        """Take another entity definition and extend this definition with its data.
        
        Returns true if any changes are made to the entity.
        """
        # Partially based on HA FGD import
        # Note: assuming classname, type, kv_order, and is_alias don't change 
        # TODO: Implement support for resources?

        has_changes = False

        # Directly overwrite the description if it's not empty
        if len(other.desc) > 0:
            self.desc = other.desc
            has_changes = True

        # Merge bases. We just combine overall...
        for base in other.bases:
            if base == self.classname:
                continue
            if base not in self.bases:
                self.bases.append(base)
                has_changes = True

        # Merge helpers. We just combine overall...
        for helper in other.helpers:
            # Sorta ew, quadratic search. But helper sizes shouldn't get too big.
            if helper not in self.helpers:
                self.helpers.append(helper)
                has_changes = True

        # Directly copy over new keyvalues, inputs, and outputs
        for cat in ('keyvalues', 'inputs', 'outputs'):
            self_map: dict[str, dict[frozenset[str], EntityDef]] = getattr(self, cat)
            other_map: dict[str, dict[frozenset[str], EntityDef]] = getattr(other, cat)
            for name, tag_map in other_map.items():
                self_map[name] = other_map[name]
                has_changes = True # Just assuming they're different in some way or another 

        return has_changes


class FGD:
    """A FGD set for a game. May be composed of several files."""
    # List of names we have already parsed.
    # We don't parse them again, to prevent infinite loops.
    _parse_list: set[File]
    # Entity definitions
    entities: dict[str, EntityDef]
    # Maximum bounding box of map
    map_size_min: int
    map_size_max: int

    # Directories we have excluded.
    mat_exclusions: set[PurePosixPath]
    # Additional dirs restricted to specific engines with tags.
    tagged_mat_exclusions: dict[TagsSet, set[PurePosixPath]]

    # Automatic visgroups.
    # The way Valve implemented this is rather strange, so we need to match
    # their data structure really to get good results. Despite it appearing
    # hierarchical in editor, we and Hammer store it flattened. Each visgroup
    # has a parent (or None for auto), and then a list of the ents it contains.
    auto_visgroups: dict[str, AutoVisgroup]

    # Snippets are named sections of syntax that can be reused.
    # Each is identified by a source filename, and a lookup key.
    snippet_desc: SnippetDict[str]
    snippet_choices: SnippetDict[Sequence[Choices]]
    snippet_flags: SnippetDict[Sequence[SpawnFlags]]
    snippet_input: SnippetDict[tuple[TagsSet, IODef]]
    snippet_output: SnippetDict[tuple[TagsSet, IODef]]
    snippet_keyvalue: SnippetDict[tuple[TagsSet, KVDef]]

    def __init__(self) -> None:
        """Create a FGD."""
        self._parse_list = set()
        self.entities = {}
        self.map_size_min = self.map_size_max = 0
        self.mat_exclusions = set()
        self.tagged_mat_exclusions = defaultdict(set)
        self.auto_visgroups = {}

        self.snippet_desc = {}
        self.snippet_choices = {}
        self.snippet_flags = {}
        self.snippet_input = {}
        self.snippet_output = {}
        self.snippet_keyvalue = {}

    @classmethod
    def parse(
        cls,
        file: Union[File[Any], str],
        filesystem: Optional[FileSystem[Any]] = None,
    ) -> FGD:
        """Parse an FGD file.

        :param file: A :py:class:filesystem.File` representing the file to read, or a file path.
        :param filesystem: The system to lookup files in. This is needed to resolve file inclusions.
            If not passed, file must be a :py:class:filesystem.File` to retrieve this automatically.
        """
        if filesystem is not None and not isinstance(file, File):
            if not file.endswith('.fgd'):
                file += '.fgd'
            try:
                file = filesystem[file]
            except KeyError:
                raise FileNotFoundError(file) from None
        elif isinstance(file, File):
            filesystem = file.sys
        else:
            raise TypeError(f'String file path passed ({file!r}), but no filesystem!')
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
                    new_bases.append(self[base])
                except KeyError:
                    raise ValueError(
                        f'Unknown base ({base}) for {ent.classname}'
                    ) from None

    def sorted_ents(self) -> Iterator[EntityDef]:
        """Yield all entities in sorted order.

        This ensures only all bases for an entity are yielded before the entity.
        Otherwise, entities are ordered in alphabetical order.
        """
        # We need to do a topological sort.
        todo: set[EntityDef] = set(self)
        done: set[EntityDef] = set()
        cls_getter = operator.attrgetter('classname')
        while todo:
            deferred: set[EntityDef] = set()
            batch = []
            for ent in todo:
                ready = True
                for base in ent.bases:
                    if isinstance(base, EntityDef) and base not in done:
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
                    f"Loop in bases! \n Problematic entities:\n"
                    f"{[ent.classname for ent in deferred]}"
                )

            todo = deferred.difference(done)

    def collapse_bases(self, ignore_aliases: bool = True) -> None:
        """Collapse all bases into the entities that use them.

        This operates in-place, and clears all the base attributes as a result.
        """
        # We need to do a topological sort effectively, to ensure we do
        # parents before children.
        for ent in self.sorted_ents():
            base_kv: list[str] = []
            keyvalue_names: set[str] = set(ent.kv_order)
            parent_resources: list[Resource] = []

            for base in ent.bases:
                if isinstance(base, str):
                    try:
                        base = self[base]
                    except KeyError:
                        raise ValueError(f'Unknown base ({base}) for {ent.classname}') from None
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

                parent_resources.extend(base.resources)

            ent.kv_order = base_kv + ent.kv_order
            ent.bases.clear()
            if parent_resources:
                parent_resources.extend(ent.resources)
                ent.resources = parent_resources

    @overload
    def export(
        self, file: FileWText, *,
        label_spawnflags: bool = True, custom_syntax: bool = True, old_report: bool = False,
    ) -> None: ...
    @overload
    def export(
        self, *,
        label_spawnflags: bool = True, custom_syntax: bool = True, old_report: bool = False,
    ) -> str: ...
    def export(
        self, file: Optional[FileWText] = None, *,
        label_spawnflags: bool = True,
        custom_syntax: bool = True,
        old_report: bool = False,
    ) -> Optional[str]:
        """Write out the FGD file.

        :param file: The file to write to. If `None`, the contents will be returned instead.
        :param label_spawnflags: If set, prepend ``[X]`` to each spawnflag name to indicate the numeric value.
        :param old_report: If enabled, use the legacy ``kv(*int)`` syntax for the 'report' flag.
            This is necessary for HL2's Hammer.
        :param custom_syntax: If disabled, all custom syntax like tags and ``@resources`` will be skipped.
            For tagged values, this can write out duplicate copies from different tags.
        """
        if file is None:
            string_buf = io.StringIO()
            file = string_buf
        else:
            string_buf = None

        if self.map_size_min != self.map_size_max:
            file.write(f'@mapsize({self.map_size_min}, {self.map_size_max})\n\n')

        if self.mat_exclusions:
            file.write('@MaterialExclusion\n\t[\n')
            for folder in sorted(self.mat_exclusions):
                file.write(f'\t"{folder!s}"\n')
            file.write('\t]\n\n')
        for tag in sorted(self.tagged_mat_exclusions):
            file.write(f'@MaterialExclusion({", ".join(sorted(tag))})\n\t[\n')
            for folder in sorted(self.tagged_mat_exclusions[tag]):
                file.write(f'\t"{folder!s}"\n')
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
                file.write(f'@AutoVisgroup = "{name_casing[parent]}"\n\t[\n')
                for visgroup in sorted(vis_by_parent[parent]):
                    file.write(f'\t"{visgroup.name}"\n\t\t[\n')
                    for ent in sorted(visgroup.ents):
                        file.write(f'\t\t"{ent}"\n')
                    file.write('\t\t]\n')
                file.write('\t]\n')
                done.add(parent)

            if todo == deferred:
                # We looped without adding one. There's an invalid one or
                # a loop or something.
                raise ValueError(
                    'Cannot export visgroups, '
                    'loop present in names: ' + ','.join([
                        f'"{self.auto_visgroups[group].parent}" -> "{group}"'
                        for group in sorted(todo)
                    ])
                )
            todo = deferred

        for ent_def in self.sorted_ents():
            file.write('\n')
            ent_def.export(file, label_spawnflags, custom_syntax, old_report)

        if string_buf is not None:
            return string_buf.getvalue()
        else:
            return None

    def parse_file(
        self,
        filesys: FileSysT,
        file: File[FileSysT],
        *,
        eval_bases: bool = True,
        eval_extensions: bool = True,
        ignore_unknown_valuetype: bool = False,
        encoding: str = 'cp1252',
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
                plus_operator=True,
                string_parens=False,
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
                        raise FileNotFoundError(file) from None
                    self.parse_file(
                        filesys,
                        include,
                        eval_bases=eval_bases,
                        eval_extensions=eval_extensions,
                        ignore_unknown_valuetype=ignore_unknown_valuetype,
                        encoding=encoding,
                    )

                elif token_value == '@mapsize':
                    # Max/min map size definition
                    tokeniser.expect(Token.PAREN_OPEN)
                    try:
                        self.map_size_min = int(tokeniser.expect(Token.STRING).strip())
                        tokeniser.expect(Token.COMMA)
                        self.map_size_max = int(tokeniser.expect(Token.STRING).strip())
                        tokeniser.expect(Token.PAREN_CLOSE)
                    except ValueError:
                        raise tokeniser.error(
                            'Invalid @MapSize definition, expected @MapSize(-65536, 65536)'
                        ) from None
                elif token_value == '@materialexclusion':
                    # Material exclusion directories.

                    # Custom syntax: (tag1, tag2) after the header, then [.
                    tags: Optional[TagsSet] = None
                    for tok, tok_value in tokeniser.skipping_newlines():
                        if tok is Token.BRACK_OPEN:
                            break
                        elif tok is Token.PAREN_OPEN and tags is None:
                            tags = read_tags(tokeniser, Token.PAREN_CLOSE)
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

                elif token_value == '@snippet':
                    Snippet.parse(self, file.path, tokeniser)

                # Entity definition...
                elif token_value[:1] == '@':
                    try:
                        ent_type = EntityTypes(token_value[1:])
                    except ValueError:
                        raise tokeniser.error(
                            'Invalid Entity type "{}"!',
                            token_value[1:],
                        ) from None
                    EntityDef.parse(self, tokeniser, ent_type, eval_bases, eval_extensions, ignore_unknown_valuetype)
                else:
                    raise tokeniser.error('Bad keyword {!r}', token_value)

    @classmethod
    def engine_dbase(cls) -> FGD:
        """Load and return a database of entity keyvalues and I/O.

        This can be used to identify the kind of keys present on an entity. If you only need
        specific entities, use :py:func:`EntityDef.engine_def()` instead to avoid needing to fetch
        all the entities.
        """
        temp_fgd = FGD()
        databases = _load_engine_db()

        if len(databases) == 1:  # If there's only one there's no need to iterate again
            return deepcopy(databases[0].get_fgd())

        for dbase in databases:
            for classname_, ent in dbase.get_fgd().entities.items():
                # Prioritise early definitions
                temp_fgd.entities.setdefault(classname_, ent)

        temp_fgd.apply_bases()
        return deepcopy(temp_fgd)

    def __getitem__(self, classname: str) -> EntityDef:
        """Lookup entities by classname."""
        try:
            return self.entities[classname.casefold()]
        except KeyError:
            raise KeyError(f'No class "{classname}"!') from None

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


_GetFGDFunc: TypeAlias = Callable[[str], EntityDef]
_EMPTY_FILESYS = VirtualFileSystem(srctools.EmptyMapping)


@attrs.frozen(init=False)
class ResourceCtx:
    """Map information passed to :attr:`FileType.ENTCLASS_FUNC` functions."""
    tags: TagsSet
    fsys: FileSystem
    #: The BSP/VMF map name, like what is passed to :command:`map` in-game.
    mapname: str
    get_entdef: Callable[[str], EntityDef]

    # For use of get_resources() only.
    _functions: Mapping[str, Callable[
        [ResourceCtx, Entity],
        Iterator[Union[Resource, Entity]]
    ]]

    def __init__(
        self,
        tags: Iterable[str] = (),
        fsys: FileSystem = _EMPTY_FILESYS,
        fgd: Union[FGD, Mapping[str, EntityDef], _GetFGDFunc] = EntityDef.engine_def,
        mapname: str = '',
        funcs: Mapping[str, Callable[
            [ResourceCtx, Entity],
            Iterator[Union[Resource, Entity]]
        ]] = srctools.EmptyMapping,
    ) -> None:
        """
        :param fgd: Used to look up dependent entities. May either be the :py:class:`FGD` itself, \
        an equivalent :external:term:`mapping`, or a callable returning the :py:class:`EntityDef`.
        If unset the internal database will be used.
        :param tags: Various string tags used to indicate what engine branch is being used. This \
        allows handling Episodic differences, enhancements by Mapbase, and other things like that.
        :param fsys: A :py:class:`~srctools.FileSystem`, used to read scripts and other files.
        :param mapname: The name of the map, used to handle some entities that used this to pick variants.
        :param funcs: Mapping of names to entclass functions to call. A builtin set of functions is
        accessed, if not present in this.
        """
        from srctools._class_resources import CLASS_FUNCS
        if funcs is srctools.EmptyMapping:
            funcs = CLASS_FUNCS
        else:
            # ChainMap itself is mutable and so can't accept Mapping.
            # We're immediately casting to Mapping, so it's not dangerous.
            funcs = ChainMap(funcs, CLASS_FUNCS)  # type: ignore[arg-type]

        # Strip extension, and normalise folder separators.
        if mapname.casefold().endswith(('.bsp', '.vmf', '.vmm', '.vmx')):
            mapname = mapname[:-4]
        self.__attrs_init__(  # pyright: ignore
            frozenset({tag.upper() for tag in tags}),
            fsys,
            mapname.replace('\\', '/'),
            # If this is an FGD or Mapping __getitem__ is the appropriate callable, otherwise
            # it must already be callable.
            getattr(fgd, '__getitem__', cast(_GetFGDFunc, fgd)),
            funcs,
        )


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
                f'Missing helper implementation for {helper}! : {HELPER_IMPL}'
            )


class _EngineDBProto(Protocol):
    """Unserialised database, which will be parsed progressively as required."""
    def get_classnames(self) -> AbstractSet[str]:
        """Get the classnames in the database."""
        raise NotImplementedError

    def get_ent(self, classname: str) -> EntityDef:
        """Fetch the specified entity."""
        raise NotImplementedError

    def get_fgd(self) -> FGD:
        """Parse all the blocks and make an FGD."""
        raise NotImplementedError

    def stats(self) -> str:
        """Return usage statistics for the database."""
        raise NotImplementedError


# Each helper type -> the class implementing them.
HELPER_IMPL: dict[HelperTypes, type[Helper]] = {}

# If we're importing, make sure _fgd_helpers is imported fresh. Otherwise, if the module is
# reloaded it'll be using the old classes, breaking our registration.
try:
    del sys.modules['srctools._fgd_helpers']
    delattr(srctools, '_fgd_helpers')  # No static analysis of this.
except (KeyError, AttributeError):
    pass

_init_helper_impl()
del _init_helper_impl
# Once done, import all the classes.
# noinspection PyProtectedMember
from srctools._fgd_helpers import *


if TYPE_CHECKING:
    KeyValues = KVDef
else:
    def __getattr__(name: str) -> type:
        """Deprecate this lookup."""
        if name == 'KeyValues':
            warnings.warn(
                'srctools.fgd.KeyValues is renamed to srctools.fgd.KVDef',
                DeprecationWarning,
                stacklevel=2,
            )
            return KVDef
        raise AttributeError(name)

    # Hide from static analysis, we want to enable this for * imports.
    globals()['__all__'].insert(6, 'KeyValues')
