"""Accessors for a builtin copy of the HammerAddons FGD database.

This lists keyvalue/io types and names available for every entity classname.
The dump does not contain help descriptions to keep the data small.
"""
from typing import (
    IO, TYPE_CHECKING, Callable, Collection, Dict, FrozenSet, List, Optional, Tuple,
)
from typing_extensions import Final
from enum import IntFlag
from struct import Struct
import io
import math

from .const import FileType
from .fgd import FGD, EntityDef, EntityTypes, IODef, KVDef, Resource, ValueTypes


__all__ = [
    'serialise', 'unserialise',
]

_fmt_8bit: Final = Struct('>B')
_fmt_16bit: Final = Struct('>H')
_fmt_32bit: Final = Struct('>I')
_fmt_double = Struct('>d')
_fmt_header = Struct('>BddI')
_fmt_ent_header = Struct('<BBBBBBB')


# Version number for the format.
BIN_FORMAT_VERSION: Final = 6
TAG_EMPTY: Final[FrozenSet[str]] = frozenset()  # This is a singleton.


class EntFlags(IntFlag):
    """Bit layout for the entity definition."""
    # First 3 bits are the entity types.
    TYPE_BASE = 0b000
    TYPE_POINT = 0b001
    TYPE_BRUSH = 0b010
    TYPE_ROPES = 0b011
    TYPE_TRACK = 0b100
    TYPE_FILTER = 0b101
    TYPE_NPC = 0b110

    MASK_TYPE = 0b111

    IS_ALIAS = 0b1000


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

FILE_TYPE_ORDER = [
    FileType.GENERIC,
    FileType.SOUNDSCRIPT,
    FileType.GAME_SOUND,
    FileType.PARTICLE,
    FileType.PARTICLE_FILE,
    FileType.VSCRIPT_SQUIRREL,
    FileType.ENTITY,
    FileType.ENTCLASS_FUNC,
    FileType.MATERIAL,
    FileType.TEXTURE,
    FileType.CHOREO,
    FileType.MODEL,
    FileType.BREAKABLE_CHUNK,
]

# Entity types -> bits used
ENTITY_TYPE_2_FLAG: Dict[EntityTypes, EntFlags] = {
    kind: EntFlags['TYPE_' + kind.name]
    for kind in EntityTypes
}

assert set(VALUE_TYPE_ORDER) == set(ValueTypes), \
    "Missing values: " + repr(set(ValueTypes) - set(VALUE_TYPE_ORDER))
assert set(ENTITY_TYPE_2_FLAG) == set(EntityTypes), \
    "Missing entity types: " + repr(set(EntityTypes) - set(ENTITY_TYPE_2_FLAG))
assert set(FILE_TYPE_ORDER) == set(FileType), \
    "Missing file types: " + repr(set(FileType) - set(FILE_TYPE_ORDER))

# Can only store this many in the bytes.
assert len(VALUE_TYPE_ORDER) < 127, "Too many values."
assert len(FILE_TYPE_ORDER) < 127, "Too many file types."

VALUE_TYPE_INDEX = {val: ind for (ind, val) in enumerate(VALUE_TYPE_ORDER)}
FILE_TYPE_INDEX = {val: ind for (ind, val) in enumerate(FILE_TYPE_ORDER)}
ENTITY_FLAG_2_TYPE = {flag: kind for (kind, flag) in ENTITY_TYPE_2_FLAG.items()}


def make_lookup(file: IO[bytes], inv_list: List[str]) -> Callable[[], str]:
    """Return a function that reads the index from the file, and returns the string it matches."""
    def lookup() -> str:
        """Perform the lookup."""
        index: int
        [index] = _fmt_16bit.unpack(file.read(2))
        return inv_list[index]
    return lookup


_py_make_lookup = _cy_make_lookup = make_lookup

# This is called a huge number of times, replace with a Cythonized version.
if not TYPE_CHECKING:
    try:
        from srctools._tokenizer import _EngineStringTable as make_lookup  # noqa
    except ImportError:
        pass
    else:
        _cy_make_lookup = make_lookup


class BinStrDict:
    """Manages a "dictionary" for compressing repeated strings in the binary format.

    Each unique string is assigned a 2-byte index into the list.
    """
    SEP = '\x1F'  # UNIT SEPARATOR

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
            assert self.SEP not in txt, repr(txt)
            inv_list[ind] = txt

        # Write it as one massive chunk.
        data = self.SEP.join(inv_list).encode('utf8')
        print(f'Dict count: {len(self._dict):,} = {len(self._dict) / (1 << 16):%}')
        print(f'Dict size: {len(data):,} bytes = {len(data) / (1 << 32):%}')
        file.write(_fmt_32bit.pack(len(data)))
        file.write(data)

    @classmethod
    def unserialise(cls, file: IO[bytes]) -> Callable[[], str]:
        """Read the dictionary from a file.

        This returns a function which reads
        a string from a file at the current point.
        """
        [length] = _fmt_32bit.unpack(file.read(4))
        inv_list = file.read(length).decode('utf8').split(cls.SEP)
        return make_lookup(file, inv_list)

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


def kv_serialise(self: KVDef, file: IO[bytes], str_dict: BinStrDict) -> None:
    """Write keyvalues to the binary file."""
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
            if tags:
                raise ValueError('Cannot use tags!')
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
        raise ValueError('CHOICES may not be used for keyvalues!')


def kv_unserialise(
    file: IO[bytes],
    from_dict: Callable[[], str],
) -> KVDef:
    """Recover a KeyValue from a binary file."""
    name = from_dict()
    disp_name = from_dict()
    [value_ind] = file.read(1)
    readonly = value_ind & 128 != 0
    value_type = VALUE_TYPE_ORDER[value_ind & 127]

    val_list: Optional[List[Tuple[int, str, bool, FrozenSet[str]]]]

    if value_type is ValueTypes.SPAWNFLAGS:
        default = ''  # No default for this type.
        [val_count] = file.read(1)
        val_list = []
        for _ in range(val_count):
            [power] = file.read(1)
            val_name = from_dict()
            val_list.append((
                1 << (power & 127),  # All flags are powers of 2.
                val_name,
                (power & 128) != 0,  # Defaults to true/false.
                TAG_EMPTY,
            ))
    else:
        default = from_dict()
        val_list = None

    # Bypass __init__, to speed up - we have a lot of these.
    kv = KVDef.__new__(KVDef)
    kv.name = name
    kv.type = value_type
    kv.disp_name = disp_name
    kv.default = default
    kv.desc = ''
    kv.val_list = val_list
    kv.readonly = readonly
    kv.reportable = False
    return kv


def iodef_serialise(iodef: IODef, file: IO[bytes], dic: BinStrDict) -> None:
    """Write an IO def the binary file."""
    file.write(dic(iodef.name))
    file.write(_fmt_8bit.pack(VALUE_TYPE_INDEX[iodef.type]))


def iodef_unserialise(
    file: IO[bytes],
    from_dict: Callable[[], str],
) -> IODef:
    """Recover an IODef from a binary file."""
    # Bypass __init__, to speed up - we have a lot of these.
    iodef = IODef.__new__(IODef)
    iodef.name = from_dict()
    [value_type] = file.read(1)
    iodef.type = VALUE_TYPE_ORDER[value_type]
    iodef.desc = ''
    return iodef


def ent_serialise(self: EntityDef, file: IO[bytes], str_dict: BinStrDict) -> None:
    """Write an entity to the binary file."""
    flags = ENTITY_TYPE_2_FLAG[self.type]
    if self.is_alias:
        flags |= EntFlags.IS_ALIAS

    file.write(_fmt_ent_header.pack(
        flags.value,
        len(self.bases),
        len(self.keyvalues),
        len(self.inputs),
        len(self.outputs),
        len(self.resources),
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
        for name, tag_map in obj_type.items():
            # We don't need to write the name, since that's stored
            # also in the kv/io object itself.

            if not tag_map:
                # No need to add this one.
                continue

            # We only support untagged things.
            if len(tag_map) == 1:
                [(tags, value)] = tag_map.items()
                if not tags:
                    if isinstance(value, KVDef):
                        kv_serialise(value, file, str_dict)
                    elif isinstance(value, IODef):
                        iodef_serialise(value, file, str_dict)
                    else:
                        raise AssertionError(f'Unknown ent attribute: {value}')
                    continue
            raise ValueError(f'{self.classname}.{name} has tags: {list(tag_map)}')

    # Helpers are not added.

    for res in self.resources:
        if res.tags:  # Tags are fairly rare.
            file.write(_fmt_8bit.pack(FILE_TYPE_INDEX[res.type] | 128))
            str_dict.write_tags(file, str_dict, res.tags)
        else:
            file.write(_fmt_8bit.pack(FILE_TYPE_INDEX[res.type]))
        file.write(str_dict(res.filename))


def ent_unserialise(
    file: IO[bytes],
    from_dict: Callable[[], str],
) -> EntityDef:
    """Read from the binary file."""
    [
        flags,
        base_count,
        kv_count,
        inp_count,
        out_count,
        res_count,
        clsname_length,
    ] = _fmt_ent_header.unpack(file.read(_fmt_ent_header.size))

    ent = EntityDef(ENTITY_FLAG_2_TYPE[flags & EntFlags.MASK_TYPE])
    ent.classname = file.read(clsname_length).decode('utf8')
    ent.desc = ''

    for _ in range(base_count):
        # We temporarily store strings, then evaluate later on.
        ent.bases.append(from_dict())

    for _ in range(kv_count):
        kv = kv_unserialise(file, from_dict)
        ent.keyvalues[kv.name] = {TAG_EMPTY: kv}
    for _ in range(inp_count):
        iodef = iodef_unserialise(file, from_dict)
        ent.inputs[iodef.name] = {TAG_EMPTY: iodef}
    for _ in range(out_count):
        iodef = iodef_unserialise(file, from_dict)
        ent.outputs[iodef.name] = {TAG_EMPTY: iodef}

    if res_count:
        resources: List[Resource] = []
        for _ in range(res_count):
            [file_ind] = file.read(1)
            file_type = FILE_TYPE_ORDER[file_ind & 127]
            if file_ind & 128:  # Has tags.
                tag = BinStrDict.read_tags(file, from_dict)
            else:
                tag = frozenset()
            resources.append(Resource(from_dict(), file_type, tag))
        ent.resources = resources

    return ent


def serialise(fgd: FGD, file: IO[bytes]) -> None:
    """Write the FGD into a compacted binary format."""
    for ent in list(fgd):
        # noinspection PyProtectedMember
        fgd._fix_missing_bases(ent)

    # The start of a file is a list of all used strings.
    dictionary = BinStrDict()

    # Start of file - format version, FGD min/max, number of entities.
    file.write(b'FGD' + _fmt_header.pack(
        BIN_FORMAT_VERSION,
        fgd.map_size_min,
        fgd.map_size_max,
        len(fgd.entities),
    ))

    ent_data = io.BytesIO()
    for ent in fgd.entities.values():
        ent_serialise(ent, ent_data, dictionary)

    # The final file is the header, dictionary data, and all the entities
    # one after each other.
    dictionary.serialise(file)
    file.write(ent_data.getvalue())


def unserialise(file: IO[bytes]) -> FGD:
    """Unpack data from engine_make_dump() to return the original data."""

    if file.read(3) != b'FGD':
        raise ValueError('Not an FGD database file!')

    fgd = FGD()

    [
        format_version,
        fgd.map_size_min,
        fgd.map_size_max,
        ent_count,
    ] = _fmt_header.unpack(file.read(_fmt_header.size))

    if format_version != BIN_FORMAT_VERSION:
        raise TypeError(f'Unknown format version "{format_version}"!')

    from_dict = BinStrDict.unserialise(file)

    # Now there's ent_count entities after each other.
    for _ in range(ent_count):
        ent = ent_unserialise(file, from_dict)
        fgd.entities[ent.classname.casefold()] = ent

    fgd.apply_bases()

    return fgd
