"""Accessors for a builtin copy of the HammerAddons FGD database.

This lists keyvalue/io types and names available for every entity classname.
The dump does not contain help descriptions to keep the data small.
"""
from typing import TYPE_CHECKING, Final, Optional, Union
from typing_extensions import TypeAlias
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Set as AbstractSet
from enum import IntFlag
from struct import Struct
import copy
import io
import itertools
import lzma
import math
import operator
import warnings

from .binformat import DeferredWrites
from .const import FileType
from .types import FileR, FileRSeek, FileWBinary, FileWBinarySeek
from .fgd import FGD, EntAttribute, EntityDef, EntityTypes, IODef, KVDef, Resource, ValueTypes


from .fgd import _EngineDBProto, _EntityView  # isort: split # noqa


__all__ = ['serialise', 'unserialise']

_fmt_8bit: Final = Struct('<B')
_fmt_dbl_8bit: Final = Struct('<BB')
_fmt_16bit: Final = Struct('<H')
_fmt_32bit: Final = Struct('<I')
_fmt_double: Final = Struct('<d')
_fmt_header: Final = Struct('<BI')
_fmt_ent_header: Final = Struct('<BBBBBB')
_fmt_block_pos: Final = Struct('<IH')


# Version number for the format.
BIN_FORMAT_VERSION: Final = 10
TAG_EMPTY: Final[frozenset[str]] = frozenset()  # This is a singleton.
# Soft limit on the number of bytes for each block, needs tuning.
MAX_BLOCK_SIZE: Final = 2048
# When writing arrays of strings, it's much more efficient to read the whole thing, decode then
# split by a character rather than read sizes individually.
STRING_SEP: Final = '\x1F'  # UNIT SEPARATOR
# Number of strings to keep in the shared database.
SHARED_STRINGS: Final = 512


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
    TYPE_EXTEND = 0b111

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
    ValueTypes.EXT_SOUNDSCAPE,
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
    FileType.WEAPON_SCRIPT,
]

# Entity types -> bits used
ENTITY_TYPE_2_FLAG: dict[EntityTypes, EntFlags] = {
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
assert len(VALUE_TYPE_ORDER) < 256, "Too many values."
assert len(FILE_TYPE_ORDER) < 127, "Too many file types."

VALUE_TYPE_INDEX = {val: ind for (ind, val) in enumerate(VALUE_TYPE_ORDER)}
FILE_TYPE_INDEX = {val: ind for (ind, val) in enumerate(FILE_TYPE_ORDER)}
ENTITY_FLAG_2_TYPE = {flag: kind for (kind, flag) in ENTITY_TYPE_2_FLAG.items()}

KV_FLAG_READONLY = 0b001
KV_FLAG_EDITOR = 0b010
KV_FLAG_REPORT = 0b100


def make_lookup(file: FileR[bytes], inv_list: list[str]) -> Callable[[], str]:
    """Return a function that reads the index from the file, and returns the string it matches."""
    def lookup() -> str:
        """Perform the lookup."""
        index: int
        [index] = _fmt_16bit.unpack(file.read(2))
        try:
            return inv_list[index]
        except IndexError:
            raise IndexError(f'Lookup {index} in {inv_list}') from None
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
    def __init__(self, database: Iterable[str], base: Optional['BinStrDict']) -> None:
        self._dict: dict[str, int] = {
            name: ind for ind, name
            in enumerate(sorted(database))
        }
        # If no base dict, this is for CBaseEntity, so set it to the real dict,
        # so __call__() won't add SHARED_STRINGS to the index.
        self.base_dict: dict[str, int] = base._dict if base is not None else self._dict
        if len(self._dict) + len(self.base_dict) >= (1 << 16):
            raise ValueError("Too many items in dictionary!")

    def __call__(self, string: str) -> bytes:
        """Get the index for a string.

        The result is the two bytes that represent the string.
        """
        if string in self.base_dict:
            return _fmt_16bit.pack(self.base_dict[string])
        else:
            return _fmt_16bit.pack(SHARED_STRINGS + self._dict[string])

    def serialise(self, file: FileWBinary) -> None:
        """Convert this to a stream of bytes."""
        inv_list = [''] * len(self._dict)
        for txt, ind in self._dict.items():
            assert STRING_SEP not in txt, repr(txt)
            inv_list[ind] = txt

        # Write it as one massive chunk.
        data = lzma.compress(STRING_SEP.join(inv_list).encode('utf8'))
        # print(f'Dict count: {len(self._dict):,} = {len(self._dict) / (1 << 16):%}')
        # print(f'Dict size: {len(data):,} bytes = {len(data) / (1 << 16):%}')
        file.write(_fmt_16bit.pack(len(data)))
        file.write(data)

    @classmethod
    def unserialise(cls, file: FileR[bytes], base: list[str]) -> tuple[list[str], Callable[[], str]]:
        """Read the dictionary from a file.

        This returns the dict, and a function which reads
        a string from a file at the current point.
        """
        [length] = _fmt_16bit.unpack(file.read(2))
        inv_list = lzma.decompress(file.read(length)).decode('utf8').split(STRING_SEP)
        # This could branch on the index to avoid the concatenation, but this should be
        # faster, and the dict will only be around temporarily anyway.
        return inv_list, make_lookup(file, base + inv_list)

    @staticmethod
    def read_tags(file: FileR[bytes], from_dict: Callable[[], str]) -> frozenset[str]:
        """Pull tags from a BinStrDict."""
        [size] = file.read(1)
        return frozenset({
            from_dict()
            for _ in range(size)
        })

    @staticmethod
    def write_tags(
        file: FileWBinary,
        dic: 'BinStrSerialise',
        tags: Collection[str],
    ) -> None:
        """Write tags a file using the dictionary."""
        file.write(_fmt_8bit.pack(len(tags)))
        for tag in tags:
            file.write(dic(tag))


class EngineDB(_EngineDBProto):
    """Unserialised database, which will be parsed progressively as required."""
    def __init__(
        self,
        ent_map: dict[str, Union[EntityDef, int]],
        base_strings: list[str],
        unparsed: list[tuple[Iterable[str], bytes]],
    ) -> None:
        self.ent_map = ent_map
        self.unparsed = unparsed
        self.base_strings = base_strings
        self.fgd: Optional[FGD] = None

    def get_classnames(self) -> AbstractSet[str]:
        """Get the classnames in the database."""
        return self.ent_map.keys()

    def stats(self) -> str:
        """Return statistics for this database."""
        parsed_block = self.unparsed.count(((), b''))
        parsed_ents: int = sum(not isinstance(x, int) for x in self.ent_map.values())
        block_count = len(self.unparsed)
        ent_count = len(self.ent_map)
        return (
            f'blocks: {parsed_block}/{block_count} = {parsed_block / block_count:.02%}, '
            f'entities: {parsed_ents}/{ent_count} = {parsed_ents / ent_count:.02%}'
        )

    def get_ent(self, classname: str) -> EntityDef:
        """Fetch the specified entity."""
        ent_info = self.ent_map[classname.casefold()]  # Or KeyError if not present.
        if isinstance(ent_info, EntityDef):
            return ent_info
        # Otherwise, we need to parse this block.
        self._parse_block(ent_info)
        entity = self.ent_map[classname.casefold()]
        assert isinstance(entity, EntityDef), f'{classname} missing from block {ent_info}'
        return entity

    def _parse_block(self, index: int) -> None:
        """Parse the specified block."""
        classes, data = self.unparsed[index]
        if not data:
            return

        cbase_entity = self.ent_map['_cbaseentity_']
        assert isinstance(cbase_entity, EntityDef)

        apply_bases = []

        file = io.BytesIO(data)
        _, from_dict = BinStrDict.unserialise(file, self.base_strings)
        for classname in classes:
            self.ent_map[classname.casefold()] = ent = ent_unserialise(file, classname, from_dict)
            if ent.bases:
                apply_bases.append(ent)
            else:
                ent.bases.append(cbase_entity)
        # We no longer need this, reset to an empty buffer.
        self.unparsed[index] = ((), b'')
        for ent in apply_bases:
            # Apply bases. This should just be for aliases, which are likely also in this block.
            # Importantly, we've already put those in ent_map, so this won't recurse if they
            # are in our block.
            ent.bases = [
                base if isinstance(base, EntityDef) else self.get_ent(base)
                for base in ent.bases
            ]

    def get_fgd(self) -> FGD:
        """Parse all the blocks and make an FGD."""
        if self.fgd is None:
            for i, (classes, data) in enumerate(self.unparsed):
                if data:
                    self._parse_block(i)
            self.fgd = FGD()
            for clsname, ent in self.ent_map.items():
                assert isinstance(ent, EntityDef), (clsname, ent)
                self.fgd.entities[clsname] = ent
            self.fgd.apply_bases()

        return copy.deepcopy(self.fgd)


def kv_serialise(kvdef: KVDef, file: FileWBinary, str_dict: BinStrSerialise) -> None:
    """Write keyvalues to the binary file."""
    file.write(str_dict(kvdef.name))
    file.write(str_dict(kvdef.disp_name))
    file.write(_fmt_dbl_8bit.pack(
        VALUE_TYPE_INDEX[kvdef.type],

        KV_FLAG_REPORT * kvdef.reportable |
        KV_FLAG_EDITOR * kvdef.editor_only |
        KV_FLAG_READONLY * kvdef.readonly
    ))

    # Spawnflags have integer names and defaults,
    # choices has string values and no default.
    if kvdef.type is ValueTypes.SPAWNFLAGS:
        file.write(_fmt_8bit.pack(len(kvdef.flags_list)))
        # spawnflags are a 32-bit int in-game, 64 in Hammer. So 128 is more than sufficient.
        for mask, name, default, tags in kvdef.flags_list:
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

    file.write(str_dict(kvdef.default))

    if kvdef.type is ValueTypes.CHOICES:
        raise ValueError('CHOICES may not be used for keyvalues!')


def kv_unserialise(
    file: FileR[bytes],
    from_dict: Callable[[], str],
) -> KVDef:
    """Recover a KeyValue from a binary file."""
    name = from_dict()
    disp_name = from_dict()
    [value_ind, kv_flags] = file.read(2)
    value_type = VALUE_TYPE_ORDER[value_ind]

    val_list: Optional[list[tuple[int, str, bool, frozenset[str]]]]

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
    kv.readonly = KV_FLAG_READONLY & kv_flags != 0
    kv.reportable = KV_FLAG_REPORT & kv_flags != 0
    kv.editor_only = KV_FLAG_EDITOR & kv_flags != 0
    return kv


def iodef_serialise(iodef: IODef, file: FileWBinary, dic: BinStrSerialise) -> None:
    """Write an IO def the binary file."""
    file.write(dic(iodef.name))
    file.write(_fmt_8bit.pack(VALUE_TYPE_INDEX[iodef.type]))


def iodef_unserialise(
    file: FileR[bytes],
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


def ent_serialise(ent: EntityDef, file: FileWBinary, str_dict: BinStrSerialise) -> None:
    """Write an entity to the binary file."""
    flags = ENTITY_TYPE_2_FLAG[ent.type]
    if ent.is_alias:
        flags |= EntFlags.IS_ALIAS

    file.write(_fmt_ent_header.pack(
        flags.value,
        len(ent.bases),
        len(ent.keyvalues),
        len(ent.inputs),
        len(ent.outputs),
        len(ent.resources),
    ))
    for base_ent in ent.bases:
        if isinstance(base_ent, str):
            file.write(str_dict(base_ent))
        else:
            file.write(str_dict(base_ent.classname))

    for obj_type in ent._iter_attrs():
        for name, tag_map in obj_type.items():
            # We don't need to write the name, since that's stored
            # also in the kv/io object itself.

            if not tag_map:
                # No need to add this one.
                continue

            # We only support untagged things.
            if len(tag_map) == 1:
                tags: frozenset[str]
                value: EntAttribute
                [(tags, value)] = tag_map.items()
                if not tags:
                    if value.custom_type is not None:
                        warnings.warn(
                            'Custom value type used for '
                            f'{ent.classname}.{name} = {value!r}!',
                            UserWarning
                        )
                    if isinstance(value, KVDef):
                        kv_serialise(value, file, str_dict)
                    elif isinstance(value, IODef):
                        iodef_serialise(value, file, str_dict)
                    else:
                        raise AssertionError(f'Unknown ent attribute: {value}')
                    continue
            raise ValueError(f'{ent.classname}.{name} has tags: {list(tag_map)}')

    # Helpers are not added.

    for res in ent.resources:
        if res.tags:  # Tags are fairly rare.
            file.write(_fmt_8bit.pack(FILE_TYPE_INDEX[res.type] | 128))
            BinStrDict.write_tags(file, str_dict, res.tags)
        else:
            file.write(_fmt_8bit.pack(FILE_TYPE_INDEX[res.type]))
        file.write(str_dict(res.filename))


def ent_unserialise(
    file: FileR[bytes],
    classname: str,
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
    ] = _fmt_ent_header.unpack(file.read(_fmt_ent_header.size))

    ent = EntityDef.__new__(EntityDef)
    ent.type = ENTITY_FLAG_2_TYPE[flags & EntFlags.MASK_TYPE]
    ent.classname = classname
    ent.desc = ''
    ent.is_alias = (EntFlags.IS_ALIAS & flags) != 0
    # We need to manually initialise.
    ent.kv_order = []
    ent.helpers = []
    ent.kv = _EntityView(ent, 'keyvalues', 'kv')
    ent.inp = _EntityView(ent, 'inputs', 'inp')
    ent.out = _EntityView(ent, 'outputs', 'out')

    # Use while loops instead of range() - often these are empty, and this code is hot, so avoid
    # building range objects and iterators.
    ent.bases = []
    while base_count:
        # We temporarily store strings, then evaluate later on.
        ent.bases.append(from_dict())
        base_count -= 1

    ent.keyvalues = {}
    while kv_count:
        kv = kv_unserialise(file, from_dict)
        ent.keyvalues[kv.name.casefold()] = {TAG_EMPTY: kv}
        kv_count -= 1

    ent.inputs = {}
    while inp_count:
        iodef = iodef_unserialise(file, from_dict)
        ent.inputs[iodef.name.casefold()] = {TAG_EMPTY: iodef}
        inp_count -= 1

    ent.outputs = {}
    while out_count:
        iodef = iodef_unserialise(file, from_dict)
        ent.outputs[iodef.name.casefold()] = {TAG_EMPTY: iodef}
        out_count -= 1

    if res_count:
        resources: list[Resource] = []
        while res_count:
            [file_ind] = file.read(1)
            file_type = FILE_TYPE_ORDER[file_ind & 127]
            if file_ind & 128:  # Has tags.
                tag = BinStrDict.read_tags(file, from_dict)
            else:
                tag = frozenset()
            resources.append(Resource(from_dict(), file_type, tag))
            res_count -= 1
        ent.resources = resources
    else:  # If empty, store a tuple that can be shared.
        ent.resources = ()

    return ent


def compute_ent_strings(ents: Iterable[EntityDef]) -> tuple[
    Mapping[EntityDef, set[str]],
    Mapping[EntityDef, int]
]:
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

    # Disable warnings, we don't want a warning both here and when it is actually serialised.
    ent: Optional[EntityDef] = None
    with warnings.catch_warnings():
        for ent in ents:
            # We don't care about the contents, so just let it overwrite itself to save reallocating
            # a new one each time.
            dummy_file.seek(0)
            ent_to_string[ent] = ent_strings = set()
            ent_serialise(ent, dummy_file, record_strings)
            ent_to_size[ent] = dummy_file.tell()

    assert ent is not None
    assert ent.classname.casefold() == '_cbaseentity_', ent
    base_strings = ent_to_string[ent]
    print(f'{SHARED_STRINGS-len(base_strings)}/{SHARED_STRINGS} shared strings used.')

    # Find common strings, move them to the base set.
    string_counts = Counter[str]()
    for strings in ent_to_string.values():
        string_counts.update(strings)
    # Shared strings might already be in base, so break early once we hit the quota.
    # At most we'll need to add SHARED_STRINGS different items.
    for st, count in string_counts.most_common(SHARED_STRINGS):
        if len(base_strings) >= SHARED_STRINGS:
            break
        base_strings.add(st)
    for strings in ent_to_string.values():
        if strings is not base_strings:
            strings -= base_strings

    return ent_to_string, ent_to_size


def build_blocks(
    all_ents: list[EntityDef],
    ent_to_string: Mapping[EntityDef, AbstractSet[str]],
    ent_to_size: Mapping[EntityDef, int],
    overlaps: Iterable[tuple[EntityDef, EntityDef, int]],
) -> list[tuple[list[EntityDef], set[str]]]:
    """Group entities into the blocks to use."""

    class BuiltBlock:
        """Used when serialising, the data."""
        def __init__(self) -> None:
            self.ents: list[EntityDef] = []
            self.stringdb: set[str] = set()
            self.bytesize: int = 0

        def add_ent(self, ent: EntityDef) -> None:
            """Add an ent to the block."""
            self.ents.append(ent)
            self.stringdb |= ent_to_string[ent]
            self.bytesize += ent_to_size[ent]
            # Circular definitions here.
            ent_to_block[ent] = self  # noqa: F821
            todo.discard(ent)  # noqa: F821

    ent_to_block: dict[EntityDef, BuiltBlock] = {}
    overflow_block = BuiltBlock()
    all_blocks: list[BuiltBlock] = [overflow_block]
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
            block = BuiltBlock()
            all_blocks.append(block)
            block.add_ent(ent1)
            block.add_ent(ent2)
    if not overflow_block.ents:
        all_blocks.remove(overflow_block)

    # Now, add every remaining ent to overflow blocks.
    print(f'{len(todo)} ents in overflow blocks.')
    for ent in list(todo):
        overflow_block.add_ent(ent)
        if overflow_block.bytesize >= MAX_BLOCK_SIZE:
            overflow_block = BuiltBlock()
            all_blocks.append(overflow_block)

    del ent_to_block, todo  # Not useful any more.
    all_blocks.sort(key=lambda block: len(block.ents))

    for block in all_blocks:
        if block.stringdb:
            efficency = format(
                sum(map(len, map(ent_to_string.__getitem__, block.ents)))
                / len(block.stringdb),
                '.02%'
            )
        else:
            efficency = 'All shared'
        print(f'{block.bytesize} bytes = {len(block.ents)} = {efficency}')
    print(len(all_blocks), 'blocks')
    return [
        (block.ents, block.stringdb)
        for block in all_blocks
    ]


def serialise(fgd: FGD, file: FileWBinarySeek) -> None:
    """Write the FGD into a compacted binary format.

    This is expected to be in engine format - _CBaseEntity_ is present, with all others based on it,
    and no other base entities.
    """
    CBaseEntity = fgd.entities.pop('_cbaseentity_')
    all_ents: list[EntityDef] = list(fgd)

    print('Computing string sizes...')
    # We need the database for CBaseEntity, but not to include it with anything else.
    ent_to_string, ent_to_size = compute_ent_strings(itertools.chain(all_ents, [CBaseEntity]))
    base_strings = ent_to_string[CBaseEntity]

    # For every pair of entities (!), compute the number of overlapping ents.
    print('Computing overlaps...')
    overlaps = [
        (ent1, ent2, len(ent_to_string[ent1] & ent_to_string[ent2]))
        for ent1, ent2 in itertools.combinations(all_ents, 2)
        if ent1.classname <= ent2.classname
    ]

    print(f'Reordering {len(overlaps)} overlaps...')
    overlaps.sort(key=operator.itemgetter(2), reverse=True)

    print('Building blocks...')
    blocks = build_blocks(all_ents, ent_to_string, ent_to_size, overlaps)

    # These dicts are big, we don't need them any more.
    del overlaps, ent_to_string, ent_to_size

    # Finally, we can serialise the file.

    # Start of file - format version, FGD min/max, number of entities.
    file.write(b'FGD' + _fmt_header.pack(BIN_FORMAT_VERSION, len(blocks)))

    deferred = DeferredWrites(file)
    for block_ents, block_stringdb in blocks:
        block_ents.sort(key=operator.attrgetter('classname'))
        for ent in block_ents:
            assert '\x1b' not in ent.classname, ent
        # Not worth it to compress these.
        classnames = STRING_SEP.join(ent.classname for ent in block_ents).encode('utf8')
        file.write(_fmt_16bit.pack(len(classnames)))
        file.write(classnames)
        deferred.defer(('block', id(block_ents)), _fmt_block_pos, write=True)

    # First, write the base strings and CBaseEntity specially.
    assert len(base_strings) == SHARED_STRINGS, len(base_strings)
    base_dict = BinStrDict(base_strings, None)
    base_dict.serialise(file)
    ent_serialise(CBaseEntity, file, base_dict)

    # Then write each block and then each entity.
    for block_ents, block_stringdb in blocks:
        block_off = file.tell()
        dictionary = BinStrDict(block_stringdb, base_dict)
        dictionary.serialise(file)
        for ent in block_ents:
            ent_serialise(ent, file, dictionary)
        block_len = file.tell() - block_off
        deferred.set_data(('block', id(block_ents)), block_off, block_len)
    deferred.write()


def unserialise(file: FileRSeek[bytes]) -> _EngineDBProto:
    """Unpack data from engine_make_dump() to return the original data."""

    if file.read(3) != b'FGD':
        raise ValueError('Not an FGD database file!')

    [format_version, block_count] = _fmt_header.unpack(file.read(_fmt_header.size))

    if format_version != BIN_FORMAT_VERSION:
        raise TypeError(f'Unknown format version {format_version}, expected {BIN_FORMAT_VERSION}!')

    block_classnames: list[list[str]] = []
    ent_map: dict[str, Union[EntityDef, int]] = {}
    unparsed: list[tuple[Iterable[str], bytes]] = []
    positions: list[tuple[list[str], int, int]] = []

    for block_id in range(block_count):
        [cls_size] = _fmt_16bit.unpack(file.read(2))
        classnames = file.read(cls_size).decode('utf8').split(STRING_SEP)
        block_classnames.append(classnames)
        for name in classnames:
            ent_map[name.casefold()] = block_id
        off, size = _fmt_block_pos.unpack(file.read(_fmt_block_pos.size))
        positions.append((classnames, off, size))

    # Read CBaseEntity.
    base_strings, from_dict = BinStrDict.unserialise(file, [])
    ent_map['_cbaseentity_'] = ent_unserialise(file, '_CBaseEntity_', from_dict)

    for classnames, off, size in positions:
        file.seek(off)
        unparsed.append((classnames, file.read(size)))

    return EngineDB(ent_map, base_strings, unparsed)
