"""Accessors for a builtin copy of the HammerAddons FGD database.

This lists keyvalue/io types and names available for every entity classname.
The dump does not contain help descriptions to keep the data small.
"""
import itertools
import operator
from typing import (
    AbstractSet, IO, Iterable, Mapping, Set, TYPE_CHECKING, Callable, Collection, Dict, FrozenSet,
    List,
    Optional,
    Tuple,
)
from typing_extensions import Final, TypeAlias
from enum import IntFlag
from struct import Struct
import io
import math

from .binformat import DeferredWrites
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
BIN_FORMAT_VERSION: Final = 7
TAG_EMPTY: Final[FrozenSet[str]] = frozenset()  # This is a singleton.
# Soft limit on the number of bytes for each block, needs tuning.
MAX_BLOCK_SIZE: Final = 2048
# When writing arrays of strings, it's much more efficient to read the whole thing, decode then
# split by a character rather than read sizes individually.
STRING_SEP: Final = '\x1F'  # UNIT SEPARATOR


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


BinStrSerialise: TypeAlias = Callable[[str], bytes]


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
            assert STRING_SEP not in txt, repr(txt)
            inv_list[ind] = txt

        # Write it as one massive chunk.
        data = STRING_SEP.join(inv_list).encode('utf8')
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
        dic: 'BinStrSerialise',
        tags: Collection[str],
    ) -> None:
        """Write tags a file using the dictionary."""
        file.write(_fmt_8bit.pack(len(tags)))
        for tag in tags:
            file.write(dic(tag))


def kv_serialise(self: KVDef, file: IO[bytes], str_dict: BinStrSerialise) -> None:
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


def iodef_serialise(iodef: IODef, file: IO[bytes], dic: BinStrSerialise) -> None:
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


def ent_serialise(self: EntityDef, file: IO[bytes], str_dict: BinStrSerialise) -> None:
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
            BinStrDict.write_tags(file, str_dict, res.tags)
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


def compute_ent_strings(ents: List[EntityDef]) -> Tuple[Mapping[EntityDef, AbstractSet[str]], Mapping[EntityDef, int]]:
    """Compute the strings each entity needs for unserialisation.

    This is done by serialising to a dummy file, noting the strings written.
    """
    dummy_file = io.BytesIO()
    ent_strings: set[str]
    ent_to_string: dict[EntityDef, set[str]] = {}
    ent_to_size: dict[EntityDef, int] = {}

    def record_strings(string: str) -> bytes:
        """Store the strings written for this entity."""
        ent_strings.add(string)
        return b'\x00\x00'

    for ent in ents:
        # We don't care about the contents, so just let it overwrite itself to save reallocating
        # a new one each time.
        dummy_file.seek(0)
        ent_to_string[ent] = ent_strings = set()
        ent_serialise(ent,dummy_file, record_strings)
        ent_to_size[ent] = dummy_file.tell()
    return ent_to_string, ent_to_size


def build_blocks(
    all_ents: List[EntityDef],
    ent_to_string: Mapping[EntityDef, AbstractSet[str]],
    ent_to_size: Mapping[EntityDef, int],
    overlaps: Iterable[Tuple[EntityDef, EntityDef, int]],
) -> List[Tuple[List[EntityDef], set[str]]]:
    """Group entities into the blocks to use."""
    class BuiltBlock:
        """Used when serialising, the data."""
        def __init__(self) -> None:
            self.ents: List[EntityDef] = []
            self.stringdb: Set[str] = set()
            self.bytesize: int = 0

        def add_ent(self, ent: EntityDef) -> None:
            """Add an ent to the block."""
            self.ents.append(ent)
            self.stringdb |= ent_to_string[ent]
            self.bytesize += ent_to_size[ent]
            ent_to_block[ent] = self
            todo.discard(ent)

    ent_to_block: dict[EntityDef, BuiltBlock] = {}
    overflow_block = BuiltBlock()
    all_blocks: List[BuiltBlock] = [overflow_block]
    todo = set(all_ents)

    for ent1, ent2, overlap_size in overlaps:
        ent1_block = ent_to_block.get(ent1)
        ent2_block = ent_to_block.get(ent2)
        if ent1_block is not None and ent2_block is not None:
            # Already allocated both, see if merging would be good.
            if ent1_block is not ent2_block and ent1_block.bytesize + ent2_block.bytesize <= MAX_BLOCK_SIZE:
                small, large = ent1_block, ent2_block
                if len(small.ents) > len(large.ents):
                    large, small = small, large
                all_blocks.remove(small)
                for ent in small.ents:
                    large.add_ent(ent)
        elif ent1_block is not None:
            if ent1_block.bytesize + ent_to_size[ent2] < MAX_BLOCK_SIZE:
                ent1_block.add_ent(ent2)
            else:
                continue  # Hope it's added by another pair.
        elif ent2_block is not None:
            if ent2_block.bytesize + ent_to_size[ent1] < MAX_BLOCK_SIZE:
                ent2_block.add_ent(ent1)
            else:
                continue  # Hope it's added by another pair.
        else:
            # Neither in a block, put both in a new block together.
            all_blocks.append(block := BuiltBlock())
            block.add_ent(ent1)
            block.add_ent(ent2)
    if not overflow_block.ents:
        all_blocks.remove(overflow_block)

    # Now, add every remaining ent to overflow blocks.
    print(f'{len(todo)} ents in overflow blocks.')
    for ent in todo:
        overflow_block.add_ent(ent)
        if overflow_block.bytesize >= MAX_BLOCK_SIZE:
            all_blocks.append(overflow_block := BuiltBlock())

    del ent_to_block, todo  # Not useful any more.
    all_blocks.sort(key=lambda block: len(block.ents))

    for block in all_blocks:
        efficency = len(block.stringdb) / sum(map(len, map(ent_to_string.__getitem__, block.ents)))
        print(f'{block.bytesize} bytes = {len(block.ents)} = {1/efficency:.02%}')
    print(len(all_blocks), 'blocks')
    return [
        (block.ents, block.stringdb)
        for block in all_blocks
    ]


def serialise(fgd: FGD, file: IO[bytes]) -> None:
    """Write the FGD into a compacted binary format.

    This is expected to be in engine format - _CBaseEntity_ is present, with all others based on it,
    and no other base entities.
    """
    CBaseEntity = fgd.entities.pop('_cbaseentity_')
    all_ents: List[EntityDef] = list(fgd)

    for ent in all_ents:
        try:
            ent.bases.remove(CBaseEntity)
        except IndexError:
            pass

    print('Computing string sizes...')
    ent_to_string, ent_to_size = compute_ent_strings(all_ents)

    # For every pair of entities (!), compute the number of overlapping ents.
    print('Computing overlaps...')
    overlaps = [
        (ent1, ent2, len(ent_to_string[ent1] & ent_to_string[ent2]))
        for ent1, ent2 in itertools.combinations(all_ents, 2)
    ]

    print('Reordering...')
    overlaps.sort(key=operator.itemgetter(2), reverse=True)

    print('Building blocks...')
    blocks = build_blocks(all_ents, ent_to_string, ent_to_size, overlaps)

    # Finally, we can serialise the file.

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
