"""Handles DataModel eXchange trees, in both binary and text (keyvalues2) format.

As an extension, optionally all strings may become full UTF-8, marked by a new
set of 'unicode_XXX' encoding formats.
"""
import struct
import sys
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Generic, NewType, KeysView,
    Dict, Tuple, Callable, IO, List, Optional, Type, MutableMapping, Iterator,
    Set, Mapping, Any, ValuesView,
)
from struct import Struct, pack
import io
import re
from uuid import UUID, uuid4 as get_uuid

from srctools import binformat, bool_as_int, BOOL_LOOKUP, Matrix, Angle
from srctools.property_parser import Property
from srctools.tokenizer import Py_Tokenizer as Tokenizer, Token


class ValueType(Enum):
    """The type of value an element has."""
    ELEMENT = 'element'  # Another attribute
    INTEGER = INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    STRING = STR = 'string'
    BINARY = BIN = VOID = 'binary'  # IE "void *", binary blob.
    TIME = 'time'  # Seconds
    COLOR = COLOUR = 'color'
    VEC2 = 'vector2'
    VEC3 = 'vector3'
    VEC4 = 'vector4'
    ANGLE = 'qangle'
    QUATERNION = 'quaternion'
    MATRIX = 'vmatrix'


# type -> enum index.
VAL_TYPE_TO_IND = {
    ValueType.ELEMENT: 1,
    ValueType.INT: 2,
    ValueType.FLOAT: 3,
    ValueType.BOOL: 4,
    ValueType.STRING: 5,
    ValueType.BINARY: 6,
    ValueType.TIME: 7,
    ValueType.COLOR: 8,
    ValueType.VEC2: 9,
    ValueType.VEC3: 10,
    ValueType.VEC4: 11,
    ValueType.ANGLE: 12,
    ValueType.QUATERNION: 13,
    ValueType.MATRIX: 14,
}
# INT_ARRAY is INT + ARRAY_OFFSET = 15, and so on.
ARRAY_OFFSET = 14
IND_TO_VALTYPE = {
    ind: val_type
    for val_type, ind in VAL_TYPE_TO_IND.items()
}
# For parsing, set this initially to check one is set.
_UNSET_UUID = get_uuid()
_UNSET = object()  # Argument sentinel
# Element type used to indicate binary "stub" elements...
STUB = '<StubElement>'


class Vec2(NamedTuple):
    """A 2-dimensional vector."""
    x: float
    y: float

    def __repr__(self) -> str:
        return f'({self[0]:.6g} {self[1]:.6g})'


class Vec3(NamedTuple):
    """A 3-dimensional vector."""
    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f'({self[0]:.6g} {self[1]:.6g} {self[2]:.6g})'


class Vec4(NamedTuple):
    """A 4-dimensional vector."""
    x: float
    y: float
    z: float
    w: float

    def __repr__(self) -> str:
        return f'({self[0]:.6g} {self[1]:.6g} {self[2]:.6g} {self[3]:.6g})'


class Quaternion(NamedTuple):
    """A quaternion used to represent rotations."""
    x: float
    y: float
    z: float
    w: float
    # def __repr__(self) -> str:
    #     return f'({self[0]} {self[1]:.6g} {self[2]:.6g} {self[3]:.6g})'


class Color(NamedTuple):
    """An RGB color."""
    r: int
    g: int
    b: int
    a: int

    def __repr__(self) -> str:
        return f'{self[0]} {self[1]} {self[2]} {self[3]}'


class AngleTup(NamedTuple):
    """A pitch-yaw-roll angle."""
    pitch: float
    yaw: float
    roll: float


Time = NewType('Time', float)
Value = Union[
    int, float, bool, str,
    bytes,
    Color,
    Time,
    Vec2, Vec3,
    Vec4,
    AngleTup,
    Quaternion,
    Matrix,
    Optional['Element'],
]

ValueT = TypeVar(
    'ValueT',
    int, float, bool, str, bytes,
    Color, Time,
    Vec2, Vec3, Vec4,
    AngleTup,
    Quaternion,
    Matrix,
    Optional['Element'],
)

# [from, to] -> conversion.
# Implementation at the end of the file.
TYPE_CONVERT: Dict[Tuple[ValueType, ValueType], Callable[[Value], Value]]
# Take valid types, convert to the value.
CONVERSIONS: Dict[ValueType, Callable[[object], Value]]
# And type -> size, excluding str/bytes.
SIZES: Dict[ValueType, int]
# Name used for keyvalues1 properties.
NAME_KV1 = 'DmElement'
# Additional name, to handle blocks with mixed properties or duplicate names.
NAME_KV1_LEAF = 'DmElementLeaf'
NAME_KV1_ROOT = 'DmElementRoot'

def parse_vector(text: str, count: int) -> List[float]:
    """Parse a space-delimited vector."""
    parts = text.split()
    if len(parts) != count:
        raise ValueError(f'{text!r} is not a {count}-dimensional vector!')
    return list(map(float, parts))


def _get_converters() -> Tuple[dict, dict, dict]:
    type_conv = {}
    convert = {}
    sizes = {}
    ns = globals()

    def unchanged(x):
        """No change means no conversion needed."""
        return x

    for from_typ in ValueType:
        for to_typ in ValueType:
            if from_typ is to_typ:
                type_conv[from_typ, to_typ] = unchanged
            else:
                func = f'_conv_{from_typ.name.casefold()}_to_{to_typ.name.casefold()}'
                try:
                    type_conv[from_typ, to_typ] = ns.pop(func)
                except KeyError:
                    if (
                        (from_typ is ValueType.STRING or to_typ is ValueType.STRING)
                        and from_typ is not ValueType.ELEMENT
                        and to_typ is not ValueType.ELEMENT
                    ):
                        raise ValueError(func + ' must exist!')
        # Special cases, variable size.
        if from_typ is not ValueType.STRING and from_typ is not ValueType.BINARY:
            sizes[from_typ] = ns['_struct_' + from_typ.name.casefold()].size
        convert[from_typ] = ns.pop(f'_conv_{from_typ.name.casefold()}')

    return type_conv, convert, sizes


def _make_val_prop(val_type: ValueType, typ: type) -> property:
    """Build the properties for each type."""

    def setter(self, value):
        self._write_val(val_type, value)

    def getter(self):
        return self._read_val(val_type)

    if val_type.name[0].casefold() in 'aeiou':
        desc = f'an {val_type.name.lower()}.'
    else:
        desc = f'a {val_type.name.lower()}.'
    getter.__doc__ = 'Return the value as ' + desc
    setter.__doc__ = 'Convert the value to ' + desc
    getter.__annotations__['return'] = typ
    setter.__annotations__['value'] = typ
    return property(
        fget=getter,
        fset=setter,
        doc='Access the value as ' + desc,
    )


class _ValProps:
    """Properties which read/write as the various kinds of value types."""

    def _read_val(self, newtype: ValueType) -> Value:
        """Convert to the desired type."""
        raise NotImplementedError

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        """Set to the desired type."""
        raise NotImplementedError

    val_int = _make_val_prop(ValueType.INT, int)
    val_str = val_string = _make_val_prop(ValueType.STRING, str)
    val_bin = val_binary = val_bytes = _make_val_prop(ValueType.BINARY, bytes)
    val_float = _make_val_prop(ValueType.FLOAT, float)
    val_time = _make_val_prop(ValueType.TIME, Time)
    val_bool = _make_val_prop(ValueType.BOOL, bool)
    val_colour = val_color = _make_val_prop(ValueType.COLOR, Color)
    val_vec2 = _make_val_prop(ValueType.VEC2, Vec2)
    val_vec3 = _make_val_prop(ValueType.VEC3, Vec3)
    val_vec4 = _make_val_prop(ValueType.VEC4, Vec4)
    val_quat = val_quaternion = _make_val_prop(ValueType.QUATERNION, Quaternion)
    val_ang = val_angle = _make_val_prop(ValueType.ANGLE, AngleTup)
    val_mat = val_matrix = _make_val_prop(ValueType.MATRIX, Matrix)
    val_compound = val_elem = val = _make_val_prop(ValueType.ELEMENT, Optional['Element'])

del _make_val_prop


# Uses private parts of Attribute only.
# noinspection PyProtectedMember
class AttrMember(_ValProps):
    """A proxy for individual indexes/keys, allowing having .val attributes."""

    def __init__(self, owner, index) -> None:
        """Internal use only."""
        self.owner = owner
        self.index = index

    def _read_val(self, newtype: ValueType) -> Value:
        if isinstance(self.owner._value, (list, dict)):
            value = self.owner._value[self.index]
        else:
            value = self.owner._value
        try:
            func = TYPE_CONVERT[self.owner._typ, newtype]
        except KeyError:
            raise ValueError(f'Cannot convert ({value}) to {newtype} type!')
        return func(value)

    def _write_val(self, newtype: ValueType, value: object) -> None:
        if newtype != self.owner._typ:
            raise ValueError('Cannot change type of array.')
        convert = CONVERSIONS[newtype](value)
        if isinstance(self.owner._value, (list, dict)):
            self.owner._value[self.index] = convert
        else:
            self.owner._value = convert


class Attribute(Generic[ValueT], _ValProps):
    """A single attribute of an element."""
    __slots__ = ['name', '_typ', '_value']
    name: str
    _typ: ValueType
    _value: Union[ValueT, List[ValueT]]

    def __init__(self, name, type, value) -> None:
        """For internal use only."""
        self.name = name
        self._typ = type
        self._value = value

    @property
    def type(self) -> ValueType:
        """Return the current type of the attribute."""
        return self._typ

    @property
    def is_array(self) -> bool:
        """Check if this is an array, or a singular value."""
        return isinstance(self._value, list)

    @classmethod
    def array(cls, name, val_type) -> 'Attribute':
        """Create an attribute with an array of a specified type."""
        return Attribute(name, val_type, [])

    @classmethod
    def int(cls, name, value) -> 'Attribute[int]':
        """Create an attribute with an integer value."""
        return Attribute(name, ValueType.INT, value)

    @classmethod
    def float(cls, name, value) -> 'Attribute[float]':
        """Create an attribute with a float value."""
        return Attribute(name, ValueType.FLOAT, value)

    @classmethod
    def time(cls, name, value) -> 'Attribute[Time]':
        """Create an attribute with a 'time' value.

        This is effectively a float, and only available in binary v3+."""
        return Attribute(name, ValueType.TIME, Time(value))

    @classmethod
    def bool(cls, name, value) -> 'Attribute[bool]':
        """Create an attribute with a boolean value."""
        return Attribute(name, ValueType.BOOL, value)

    @classmethod
    def string(cls, name, value) -> 'Attribute[str]':
        """Create an attribute with a string value."""
        return Attribute(name, ValueType.STRING, value)

    @classmethod
    def binary(cls, name, value) -> 'Attribute[bytes]':
        """Create an attribute with binary data."""
        return Attribute(name, ValueType.BINARY, value)

    @classmethod
    def vec2(cls, name, x=0.0, y=0.0) -> 'Attribute[Vec2]':
        """Create an attribute with a 2D vector."""
        if isinstance(x, (int, float)):
            x_ = float(x)
        else:
            it = iter(x)
            x_ = float(next(it, 0.0))
            y = float(next(it, y))
        return Attribute(name, ValueType.VEC2, Vec2(x_, y))

    @classmethod
    def vec3(cls, name, x=0.0, y=0.0, z=0.0) -> 'Attribute[Vec3]':
        """Create an attribute with a 3D vector."""
        if isinstance(x, (int, float)):
            x_ = float(x)
        else:
            it = iter(x)
            x_ = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
        return Attribute(name, ValueType.VEC3, Vec3(x_, y, z))

    @classmethod
    def vec4(cls, name, x=0.0, y=0.0, z=0.0, w=0.0) -> 'Attribute[Vec4]':
        """Create an attribute with a 4D vector."""
        if isinstance(x, (int, float)):
            x_ = float(x)
        else:
            it = iter(x)
            x_ = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
            w = float(next(it, w))
        return Attribute(name, ValueType.VEC4, Vec4(x_, y, z, w))

    @classmethod
    def color(cls, name, r=0, g=0, b=0, a=255) -> 'Attribute[Color]':
        """Create an attribute with a color."""
        if isinstance(r, int):
            r_ = r
        else:
            it = iter(r)
            r_ = int(next(it, 0))
            g = int(next(it, g))
            b = int(next(it, b))
            a = int(next(it, a))
        return Attribute(name, ValueType.COLOR, Color(r_, g, b, a))

    @classmethod
    def angle(cls, name, pitch=0.0, yaw=0.0, roll=0.0) -> 'Attribute[AngleTup]':
        """Create an attribute with an Euler angle."""
        if isinstance(pitch, (int, float)):
            pitch_ = float(pitch)
        else:
            it = iter(pitch)
            pitch_ = float(next(it, 0.0))
            yaw = float(next(it, yaw))
            roll = float(next(it, roll))
        return Attribute(name, ValueType.ANGLE, AngleTup(pitch_, yaw, roll))

    @classmethod
    def quaternion(cls, name: str, x=0.0, y=0.0, z=0.0, w=1.0) -> 'Attribute[Quaternion]':
        """Create an attribute with a quaternion rotation."""
        if isinstance(x, (int, float)):
            x_ = float(x)
        else:
            it = iter(x)
            x_ = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
            w = float(next(it, w))
        return Attribute(name, ValueType.QUATERNION, Quaternion(x_, y, z, w))

    def _read_val(self, newtype: ValueType) -> Value:
        """Convert to the desired type."""
        if isinstance(self._value, list):
            raise ValueError('Cannot read value of array elements!')
        if isinstance(self._value, dict):
            raise ValueError('Cannot read value of compound elements!')
        try:
            func = TYPE_CONVERT[self._typ, newtype]
        except KeyError:
            raise ValueError(
                f'Cannot convert ({self._value!r}) to {newtype} type!')
        return func(self._value)

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        """Change the type of the atribute."""
        self._typ = newtype
        self._value = CONVERSIONS[newtype](value)  # type: ignore # This changes the generic...

    def __repr__(self) -> str:
        return f'<{self._typ.name} Attr {self.name!r}: {self._value!r}>'

    def __getitem__(self, item):
        """Read values in an array element."""
        if not isinstance(self._value, list):
            raise ValueError('Cannot index singular elements.')
        _ = self._value[item]  # Raise IndexError/KeyError if not present.
        return AttrMember(self, item)

    def __setitem__(self, ind, value):
        """Set a specific array element to a value."""
        if not isinstance(self._value, list):
            raise ValueError('Cannot index singular elements.')
        [val_type, result] = deduce_type_single(value)
        if val_type is not self._typ:
            # Try converting.
            try:
                func = TYPE_CONVERT[self._typ, val_type]
            except KeyError:
                raise ValueError(
                    f'Cannot convert ({val_type}) to {self._typ} type!')
            self._value[ind] = func(result)
        else:
            self._value[ind] = result

    def __delitem__(self, item):
        """Remove the specified array index."""
        if not isinstance(self._value, list):
            raise ValueError('Cannot index singular elements.')
        del self._value[item]

    def __len__(self):
        """Return the number of values in the array, if this is one."""
        if isinstance(self._value, list):
            return len(self._value)
        raise ValueError('Singular elements have no length!')

    def __iter__(self):
        """Yield each of the elements in an array."""
        if isinstance(self._value, list):
            return iter(self._value)
        else:
            return iter((self._value, ))

    def append(self, value) -> None:
        """Append an item to the array.

        If not already an array, it is converted to one
        holding the existing value.
        """
        if not isinstance(self._value, list):
            self._value = [self._value]
        [val_type, result] = deduce_type_single(value)
        if val_type is not self._typ:
            # Try converting.
            try:
                func = TYPE_CONVERT[self._typ, val_type]
            except KeyError:
                raise ValueError(
                    f'Cannot convert ({val_type}) to {self._typ} type!')
            self._value.append(func(result))
        else:
            self._value.append(result)


class Element(MutableMapping[str, Attribute]):
    """An element in a DMX tree."""
    __slots__ = ['type', 'name', 'uuid', '_members']
    name: str
    type: str
    uuid: UUID
    _members: Dict[str, Attribute]

    def __init__(self, name: str, type: str, uuid: UUID=None) -> None:
        self.name = name
        self.type = type
        self._members = {}
        if uuid is None:
            self.uuid = get_uuid()
        else:
            self.uuid = uuid

    @classmethod
    def parse(cls, file: IO[bytes], unicode=False) -> Tuple['Element', str, int]:
        """Parse a DMX file encoded in binary or KV2 (text).

        The return value is the tree, format name and version.
        If unicode is set to True, strings will be treated as UTF8 instead
        of safe ASCII.
        """
        # The format header is:
        # <!-- dmx encoding [encoding] [version] format [format] [version] -->
        header = bytearray(file.read(4))
        if header != b'<!--':
            raise ValueError('The file is not a DMX file.')
        # Read until the -->, or we arbitrarily hit 1kb (assume it's corrupt)
        for i in range(1024):
            header.extend(file.read(1))
            if header.endswith(b'-->'):
                break
        else:
            raise ValueError('Unterminated DMX heading comment!')
        match = re.match(
            br'<!--\s*dmx\s+encoding\s+(unicode_)?(\S+)\s+([0-9]+)\s+'
            br'format\s+(\S+)\s+([0-9]+)\s*-->', header,
        )
        if match is not None:
            unicode_flag, enc_name, enc_vers_by, fmt_name_by, fmt_vers_by = match.groups()

            if unicode_flag:
                unicode = True
            enc_vers = int(enc_vers_by.decode('ascii'))
            fmt_name = fmt_name_by.decode('ascii')
            fmt_vers = int(fmt_vers_by.decode('ascii'))
        else:
            # Try a "legacy" header, no version.
            match = re.match(br'<!--\s*DMXVersion\s+([a-z0-9]+)_v[a-z0-9]*\s*-->', header)
            if match is None:
                raise ValueError(f'Invalid DMX header {bytes(header)!r}!')
            enc_name = match.group(0)
            if enc_name == b'sfm':
                enc_name = b'binary'
            unicode = False
            enc_vers = 0
            fmt_name = ''
            fmt_vers = 0

        if enc_name == b'keyvalues2':
            result = cls.parse_kv2(
                io.TextIOWrapper(file, encoding='utf8' if unicode else 'ascii'),
                enc_vers,
            )
        elif enc_name == b'binary':
            result = cls.parse_bin(file, enc_vers, unicode)
        else:
            raise ValueError(f'Unknown DMX encoding {repr(enc_name)[2:-1]}!')

        return result, fmt_name, fmt_vers

    @classmethod
    def parse_bin(cls, file, version, unicode=False):
        """Parse the core binary data in a DMX file.

        The <!-- --> format comment line should have already be read.
        If unicode is set to True, strings will be treated as UTF8 instead
        of safe ASCII.
        """
        # There should be a newline and null byte after the comment.
        newline = file.read(2)
        if newline != b'\n\0':
            raise ValueError('No newline after comment!')

        encoding = 'utf8' if unicode else 'ascii'

        # First, we read the string "dictionary".
        if version >= 5:
            stringdb_size = stringdb_ind = '<i'
        elif version >= 4:
            stringdb_size = '<i'
            stringdb_ind = '<h'
        elif version >= 2:
            stringdb_size = stringdb_ind = '<h'
        else:
            stringdb_size = stringdb_ind = None

        if stringdb_size is not None:
            [string_count] = binformat.struct_read(stringdb_size, file)
            stringdb = binformat.read_nullstr_array(file, string_count, encoding)
        else:
            stringdb = None

        stubs: Dict[UUID, Element] = {}

        [element_count] = binformat.struct_read('<i', file)
        elements: List[Element] = [None] * element_count
        for i in range(element_count):
            if stringdb is not None:
                [ind] = binformat.struct_read(stringdb_ind, file)
                el_type = stringdb[ind]
            else:
                el_type = binformat.read_nullstr(file)
            if version >= 4:
                [ind] = binformat.struct_read(stringdb_ind, file)
                name = stringdb[ind]
            else:
                name = binformat.read_nullstr(file, encoding=encoding)
            uuid = UUID(bytes_le=file.read(16))
            elements[i] = Element(name, el_type, uuid)
        # Now, the attributes in the elements.
        for i in range(element_count):
            elem = elements[i]
            [attr_count] = binformat.struct_read('<i', file)
            for attr_i in range(attr_count):
                if stringdb is not None:
                    [ind] = binformat.struct_read(stringdb_ind, file)
                    name = stringdb[ind]
                else:
                    name = binformat.read_nullstr(file, encoding=encoding)
                [attr_type_data] = binformat.struct_read('<B', file)
                array_size: Optional[int]
                if attr_type_data >= ARRAY_OFFSET:
                    attr_type_data -= ARRAY_OFFSET
                    [array_size] = binformat.struct_read('<i', file)
                else:
                    array_size = None
                attr_type = IND_TO_VALTYPE[attr_type_data]

                if attr_type is ValueType.TIME and version < 3:
                    # It's elementid in these versions ???
                    raise ValueError('Time attribute added in version 3!')
                elif attr_type is ValueType.ELEMENT:
                    if array_size is not None:
                        array = []
                        attr = Attribute(name, attr_type, array)
                        for _ in range(array_size):
                            [ind] = binformat.struct_read('<i', file)
                            if ind == -1:
                                child_elem = None
                            elif ind == -2:
                                # Stub element, just with a UUID.
                                [uuid_str] = binformat.read_nullstr(file)
                                uuid = UUID(uuid_str)
                                try:
                                    child_elem = stubs[uuid]
                                except KeyError:
                                    child_elem = stubs[uuid] = Element('', 'StubElement', uuid)
                            else:
                                child_elem = elements[ind]
                            array.append(child_elem)
                    else:
                        [ind] = binformat.struct_read('<i', file)
                        if ind == -1:
                            child_elem = None
                        elif ind == -2:
                            # Stub element, just with a UUID.
                            [uuid_str] = binformat.read_nullstr(file)
                            uuid = UUID(uuid_str)
                            try:
                                child_elem = stubs[uuid]
                            except KeyError:
                                child_elem = stubs[uuid] = Element('', 'StubElement', uuid)
                        else:
                            child_elem = elements[ind]
                        attr = Attribute(name, ValueType.ELEMENT, child_elem)
                elif attr_type is ValueType.STRING:
                    if array_size is not None:
                        # Arrays are always raw ASCII in the file.
                        attr = Attribute(
                            name, attr_type,
                            binformat.read_nullstr_array(file, array_size),
                        )
                    else:  # Single string.
                        if stringdb is not None and version >= 4:
                            [ind] = binformat.struct_read(stringdb_ind, file)
                            value = stringdb[ind]
                        else:
                            # Raw value.
                            value = binformat.read_nullstr(file, encoding=encoding)
                        attr = Attribute(name, attr_type, value)
                elif attr_type is ValueType.BINARY:
                    # Binary blobs.
                    if array_size is not None:
                        array = []
                        attr = Attribute(name, attr_type, array)
                        for _ in range(array_size):
                            [size] = binformat.struct_read('<i', file)
                            array.append(file.read(size))
                    else:
                        [size] = binformat.struct_read('<i', file)
                        attr = Attribute(name, attr_type, file.read(size))
                else:
                    # All other types are fixed-length.
                    size = SIZES[attr_type]
                    conv = TYPE_CONVERT[ValueType.BINARY, attr_type]
                    if array_size is not None:
                        attr = Attribute(name, attr_type, [
                            conv(file.read(size))
                            for _ in range(array_size)
                        ])
                    else:
                        attr = Attribute(name, attr_type, conv(file.read(size)))
                elem._members[name.casefold()] = attr

        return elements[0]

    @classmethod
    def parse_kv2(cls, file, version):
        """Parse a DMX file encoded in KeyValues2.

        The <!-- --> format comment line should have already be read.
        """
        # We apply UUID lookups after everything's parsed.
        id_to_elem: Dict[UUID, Element] = {}

        # Locations in arrays which are UUIDs (and need setting).
        # This is a (attr, index, uuid, line_num) tuple.
        fixups: List[Tuple[Attribute, Optional[int], UUID, int]] = []

        elements = []

        tok = Tokenizer(file)
        for token, tok_value in tok:
            if token is Token.STRING:
                elem_name = tok_value
            elif token is Token.NEWLINE:
                continue
            else:
                raise tok.error(token)
            elements.append(cls._parse_kv2_element(tok, id_to_elem, fixups, '', elem_name))

        for attr, index, uuid, line_num in fixups:
            try:
                elem = id_to_elem[uuid]
            except KeyError:
                tok.line_num = line_num
                raise tok.error('UUID {} not found!', uuid)
            if index is None:
                attr._value = elem
            else:
                attr._value[index] = elem

        return elements[0]

    @classmethod
    def _parse_kv2_element(cls, tok, id_to_elem, fixups, name, typ_name):
        """Parse a compound element."""
        elem: Element = cls(name, typ_name, _UNSET_UUID)
        for attr_name in tok.block(name):
            orig_typ_name = tok.expect(Token.STRING)
            typ_name = orig_typ_name.casefold()

            # The UUID is a special element name/type combo.
            if attr_name == 'id':
                if typ_name != 'elementid':
                    raise tok.error(
                        'Element ID attribute must be '
                        '"elementid" type, not "{}"!',
                        typ_name
                    )
                uuid_str = tok.expect(Token.STRING)
                if elem.uuid is not _UNSET_UUID:
                    raise tok.error('Duplicate UUID definition!')
                try:
                    elem.uuid = UUID(uuid_str)
                except ValueError:
                    raise tok.error('Invalid UUID "{}"!', uuid_str)
                id_to_elem[elem.uuid] = elem
                continue
            elif attr_name == 'name':  # This is also special.
                if typ_name != 'string':
                    raise tok.error(
                        'Element name attribute must be '
                        '"string" type, not "{}"!',
                        typ_name
                    )
                elem.name = tok.expect(Token.STRING)
                continue

            if typ_name.endswith('_array'):
                is_array = True
                typ_name = typ_name[:-6]
            else:
                is_array = False

            try:
                attr_type = ValueType(typ_name)
            except ValueError:
                # It's an inline compound element.
                elem._members[attr_name.casefold()] = Attribute(
                    attr_name, ValueType.ELEMENT,
                    cls._parse_kv2_element(tok, id_to_elem, fixups, attr_name, orig_typ_name),
                )
                continue
            if is_array:
                array = []
                attr = Attribute(attr_name, attr_type, array)
                tok.expect(Token.BRACK_OPEN)
                for tok_typ, tok_value in tok:
                    if tok_typ is Token.BRACK_CLOSE:
                        break
                    elif tok_typ is Token.STRING:
                        if attr_type is ValueType.ELEMENT:
                            if tok_value == 'element':
                                # UUID reference.
                                uuid_str = tok.expect(Token.STRING)
                                if uuid_str:
                                    try:
                                        uuid = UUID(uuid_str)
                                    except ValueError:
                                        raise tok.error('Invalid UUID "{}"!', uuid_str)
                                    fixups.append((attr, len(array), uuid, tok.line_num))
                                # If UUID is present, this None will be
                                # overwritten after. Otherwise, this stays None.
                                array.append(None)
                            else:
                                # Inline compound
                                array.append(cls._parse_kv2_element(tok, id_to_elem, fixups, attr_name, tok_value))
                        else:
                            # Other value
                            try:
                                array.append(TYPE_CONVERT[ValueType.STRING, attr_type](tok_value))
                            except ValueError:
                                raise tok.error('"{}" is not a valid {}!', tok_value, attr_type.name)
                        # Skip over the trailing comma if present.
                        next_tok, tok_value = tok()
                        while next_tok is Token.NEWLINE:
                            next_tok, tok_value = tok()
                        if next_tok is not Token.COMMA:
                            tok.push_back(next_tok, tok_value)
                    elif tok_typ is not Token.NEWLINE:
                        raise tok.error(tok_typ)
                else:
                    raise tok.error('Unterminated array!')
            elif attr_type is ValueType.ELEMENT:
                # This is a reference to another element.
                uuid_str = tok.expect(Token.STRING)
                attr = Attribute(attr_name, attr_type, None)
                if uuid_str:
                    try:
                        uuid = UUID(uuid_str)
                    except ValueError:
                        raise tok.error('Invalid UUID "{}"!', uuid_str)
                    fixups.append((attr, None, uuid, tok.line_num))
                # If UUID is present, the None value  will be overwritten after.
                # Otherwise, this stays None.
            else:
                # Single element.
                unparsed = tok.expect(Token.STRING)
                value = TYPE_CONVERT[ValueType.STRING, attr_type](unparsed)
                attr = Attribute(attr_name, attr_type, value)
            elem._members[attr_name.casefold()] = attr

        if elem.uuid is _UNSET_UUID:
            # No UUID set, just generate one.
            elem.uuid = get_uuid()
        return elem

    def export_binary(
        self, file: IO[bytes],
        version: int = 5,
        fmt_name: str = 'dmx', fmt_ver: int = 1,
        unicode: str='ascii',
    ) -> None:
        """Write out a DMX tree, using the binary format.

        The version must be a number from 0-5.
        The format name and version can be anything, to indicate which
        application should read the file.
        Unicode controls whether Unicode characters are permitted:
        - 'ascii' (the default) raises an error if any value is non-ASCII. This
          ensures no encoding issues occur when read by the game.
        - 'format' changes the encoding format to 'unicode_binary', allowing
          the file to be rejected if the game tries to read it and this module's
          parser to automatically switch to Unicode.
        - 'silent' outputs UTF8 without any marker, meaning it could be parsed
          incorrectly by the game or other utilties. This must be parsed with
          unicode=True to succeed.
        """
        if not (0 <= version <= 5):
            raise ValueError(f'Invalid version: {version} is not within range 0-5!')
        # Write the header, and determine which string DB variant to use.
        if version == 0:
            # "legacy" header.
            file.write(
                b'<!-- DMXVersion %b_v2 -->\n\0'
                % (fmt_name.encode('ascii'), )
            )
        else:
            file.write(b'<!-- dmx encoding %sbinary %i format %b %i -->\n\0' % (
                b'unicode_' if unicode == 'format' else b'',
                version,
                fmt_name.encode('ascii'),
                fmt_ver,
            ))
        if version >= 5:
            stringdb_size = stringdb_ind = '<i'
        elif version >= 4:
            stringdb_size = '<i'
            stringdb_ind = '<h'
        elif version >= 2:
            stringdb_size = stringdb_ind = '<h'
        else:
            stringdb_size = stringdb_ind = None

        encoding = 'utf8' if unicode != 'ascii' else 'ascii'

        # First, iterate the tree to collect the elements, and strings (v2+).
        elements: List[Element] = [self]
        elem_to_ind: Dict[UUID, int] = {self.uuid: 0}
        # Valve "bug" - the name attribute is in the database, despite having a
        # special location in the file format.
        used_strings: Set[str] = {"name"}

        # Use the fact that list iteration will continue through appended
        # items.
        for elem in elements:
            if stringdb_ind is not None:
                used_strings.add(elem.type)
            if version >= 4:
                used_strings.add(elem.name)
            for attr in elem.values():
                if stringdb_ind is not None:
                    used_strings.add(attr.name)
                if attr.type is ValueType.TIME and version < 3:
                    raise ValueError('TIME attributes are not permitted before binary v3!')
                elif attr.type is ValueType.ELEMENT:
                    for subelem in attr:
                        if subelem is not None and subelem.type != STUB and subelem.uuid not in elem_to_ind:
                            elem_to_ind[subelem.uuid] = len(elements)
                            elements.append(subelem)
                # Only non-array strings get added to the DB.
                elif version >= 4 and attr.type is ValueType.STRING and not attr.is_array:
                    used_strings.add(attr.val_str)

        string_list = sorted(used_strings)
        string_to_ind = {
            text: ind
            for ind, text in enumerate(string_list)
        }
        if stringdb_size is not None:
            file.write(pack(stringdb_size, len(string_list)))
            for text in string_list:
                file.write(text.encode(encoding) + b'\0')
        file.write(pack('<i', len(elements)))

        for elem in elements:
            if stringdb_ind is not None:
                file.write(pack(stringdb_ind, string_to_ind[elem.type]))
            else:
                file.write(elem.type.encode(encoding) + b'\0')
            if version >= 4:
                file.write(pack(stringdb_ind, string_to_ind[elem.name]))
            else:
                file.write(elem.name.encode(encoding) + b'\0')
            file.write(elem.uuid.bytes_le)

        # Now, write out all attributes.
        for elem in elements:
            file.write(pack('<i', len(elem)))
            for attr in elem.values():
                if stringdb_ind is not None:
                    file.write(pack(stringdb_ind, string_to_ind[attr.name]))
                else:
                    file.write(attr.name.encode(encoding) + b'\0')
                typ_ind = VAL_TYPE_TO_IND[attr.type]
                if attr.is_array:
                    typ_ind += ARRAY_OFFSET
                file.write(pack('B', typ_ind))
                if attr.is_array:
                    file.write(pack('<i', len(attr)))

                if attr.type is ValueType.STRING:
                    # Scalar strings after v4 use the DB.
                    if version >= 4 and not attr.is_array:
                        file.write(pack(stringdb_ind, string_to_ind[attr.val_str]))
                    else:
                        for text in attr:
                            file.write(text.encode(encoding) + b'\0')
                elif attr.type is ValueType.BINARY:
                    for data in attr:
                        file.write(pack('<i', len(data)))
                        file.write(data)
                elif attr.type is ValueType.ELEMENT:
                    for subelem in attr:
                        if subelem is None:
                            elm_ind = -1
                            file.write(pack('<i', -1))
                        elif subelem.type == STUB:
                            elm_ind = -2
                            file.write(pack('<i', -2))
                        else:
                            file.write(pack('<i', elem_to_ind[subelem.uuid]))
                else:
                    conv_func = TYPE_CONVERT[attr.type, ValueType.BINARY]
                    for any_data in attr:
                        try:
                            file.write(conv_func(any_data))
                        except struct.error:
                            raise ValueError(f'Cannot convert {attr.type}({any_data}) to binary!')

    def export_kv2(
        self, file: IO[bytes],
        fmt_name: str = 'dmx', fmt_ver: int = 1,
        *,
        flat: bool = False,
        unicode: str='ascii',
        cull_uuid: bool = False,
    ) -> None:
        """Write out a DMX tree, using the text-based KeyValues2 format.

        The format name and version can be anything, to indicate which
        application should read the file.

        * If flat is enabled, elements will all be placed at the toplevel,
          so they don't nest inside each other.
        * If cull_uuid is enabled, UUIDs are only written for self-referential
          elements. When parsed by this or Valve's parser, new ones will simply
          be generated.

        * unicode controls whether Unicode characters are permitted:
            - 'ascii' (the default) raises an error if any value is non-ASCII.
              This ensures no encoding issues occur when read by the game.
            - 'format' changes the encoding format to 'unicode_keyvalues2',
              allowing the file to be rejected if the game tries to read it and
              this module's parser to automatically switch to Unicode.
            - 'silent' outputs UTF8 without any marker, meaning it could be
              parsed incorrectly by the game or other utilties. This must be
              parsed with  unicode=True to succeed.
        """
        file.write(b'<!-- dmx encoding %skeyvalues2 1 format %b %i -->\r\n' % (
            b'unicode_' if unicode == 'format' else b'',
            fmt_name.encode('ascii'),
            fmt_ver,
        ))
        # First, iterate through to find all "root" elements.
        # Those are added with UUIDS, and are put at the end of the file.
        # If `flat` is enabled, all elements are like that. Otherwise,
        # it's only self and any used multiple times.
        elements: List[Element] = [self]
        use_count: Dict[UUID, int] = {self.uuid: 1}

        # Use the fact that list iteration will continue through appended items.
        for elem in elements:
            for attr in elem.values():
                if attr.type is not ValueType.ELEMENT:
                    continue
                for subelem in attr:
                    if subelem is None or subelem.type == STUB:
                        continue
                    if subelem.uuid not in use_count:
                        use_count[subelem.uuid] = 1
                        elements.append(subelem)
                    else:
                        use_count[subelem.uuid] += 1

        if flat:
            roots = set(use_count)
        else:
            roots = {uuid for uuid, count in use_count.items() if count > 1}
        # We're always a root!
        roots.add(self.uuid)

        encoding = 'utf8' if unicode != 'ascii' else 'ascii'

        for elem in elements:
            if flat or elem.uuid in roots:
                if elem is not self:
                    file.write(b'\r\n')
                elem._export_kv2(file, b'', roots, encoding, cull_uuid)
                file.write(b'\r\n')

    def _export_kv2(
        self,
        file: IO[bytes],
        indent: bytes,
        roots: Set[UUID],
        encoding: str,
        cull_uuid: bool,
    ) -> None:
        """Export a single element to the file.

        @param indent: The tabs to prepend each line with.
        @param roots: The set of all elements exported at toplevel, and so
            needs to be referenced by UUID instead of inline.
        """
        indent_child = indent + b'\t'
        file.write(b'"%b"\r\n%b{\r\n' % (self.type.encode('ascii'), indent))
        if not cull_uuid or self.uuid in roots:
            file.write(b'%b"id" "elementid" "%b"\r\n' % (indent_child, str(self.uuid).encode('ascii')))
        file.write(b'%b"name" "string" "%b"\r\n' % (indent_child, self.name.encode(encoding)))
        for attr in self.values():
            file.write(b'%b"%b" ' % (
                indent_child,
                attr.name.encode(encoding),
            ))
            if attr.is_array:
                file.write(b'"%b_array"\r\n%b[\r\n' % (attr.type.value.encode(encoding), indent_child))
                indent_arr = indent + b'\t\t'
                for i, child in enumerate(attr):
                    file.write(indent_arr)
                    if isinstance(child, Element):
                        if child.uuid in roots:
                            file.write(b'"element" "%b"' % str(child.uuid).encode('ascii'))
                        else:
                            child._export_kv2(file, indent_arr, roots, encoding, cull_uuid)
                    else:
                        str_value = TYPE_CONVERT[attr.type, ValueType.STRING](child).encode(encoding)
                        file.write(b'"%b"' % (str_value, ))
                    if i == len(attr) - 1:
                        file.write(b'\r\n')
                    else:
                        file.write(b',\r\n')
                file.write(b'%b]\r\n' % (indent_child, ))
            elif isinstance(attr._value, Element):
                if attr.val_elem.uuid in roots:
                    file.write(b'"element" "%b"\r\n' % str(attr.val_elem.uuid).encode('ascii'))
                else:
                    attr.val_elem._export_kv2(file, indent_child, roots, encoding, cull_uuid)
                    file.write(b'\r\n')
            else:
                file.write(b'"%b" "%b"\r\n' % (
                    attr.type.value.encode(encoding),
                    attr.val_str.encode(encoding),
                ))
        file.write(indent + b'}')

    @classmethod
    def from_kv1(cls, props: Property) -> 'Element':
        """Convert a KeyValues 1 property tree into DMX format.

        All blocks have a type of "DmElement", with children stored in the "subkeys" array. Leaf
        properties are stored as regular attributes. The following attributes are prepended with
        an underscore when converting as they are reserved: "name" and "subkeys".

        If multiple leaf properties with the same name or the element has a mix of blocks and leafs
        all elements will be put in the subkeys array. Leafs will use "DmElementLeaf" elements with
        values in the "value" key.
        """
        if not props.has_children():
            elem = cls(props.real_name, NAME_KV1_LEAF)
            elem['value'] = props.value
            return elem

        if props.is_root():
            elem = cls('', NAME_KV1_ROOT)
        else:
            elem = cls(props.real_name, NAME_KV1)

        # First go through to check if we can inline attributes, or have to nest.
        # If we have duplicates, both types, or any of the reserved names we need to do so.
        leaf_names: set[str] = set()
        has_leaf = False
        has_block = False
        no_inline = False
        for child in props:
            if child.has_children():
                has_block = True
            else:
                has_leaf = True
                # The names "name" and "subkeys" are reserved, and can't be used as attributes.
                # ID isn't, because it has a unique attr type to distinguish.
                if child.name in {'name', 'subkeys'}:
                    no_inline = True
                if child.name in leaf_names:
                    no_inline = True
                else:
                    leaf_names.add(child.name)
        del leaf_names
        if has_block and has_leaf:
            no_inline = True

        subkeys: Optional[Attribute[Element]] = None
        if no_inline or has_block:
            elem['subkeys'] = subkeys = Attribute.array('subkeys', ValueType.ELEMENT)

        for child in props:
            if no_inline or child.has_children():
                assert subkeys is not None
                subkeys.append(cls.from_kv1(child))
            else:
                elem[child.real_name] = child.value
        return elem

    def to_kv1(self) -> Property:
        """Convert an element tree containing a KeyValues 1 tree back into a Property.

        These must satisfy the format from_kv1() produces - all elements have the type DmElement,
        all attributes are strings except for the "subkeys" attribute which is an element array.
        """
        if self.type == NAME_KV1_LEAF:
            return Property(self.name, self['value'].val_str)
        elif self.type == NAME_KV1:
            prop = Property(self.name, [])
        elif self.type == NAME_KV1_ROOT:
            prop = Property.root()
        else:
            raise ValueError(f'{self.type}({self.name!r}) is not a KeyValues1 tree!')
        subkeys: Optional[Attribute[Element]] = None
        for attr in self.values():
            if attr.name == 'subkeys':
                if attr.type != ValueType.ELEMENT or not attr.is_array:
                    raise ValueError('"subkeys" must be an Element array!')
                subkeys = attr
            else:
                prop.append(Property(attr.name, attr.val_str))
        if subkeys is not None:
            for elem in subkeys:
                prop.append(elem.to_kv1())
        return prop

    def __repr__(self) -> str:
        if self.type:
            return f'<{self.type}({self.name!r}): {self._members!r}>'
        elif self.name:
            return f'<Element {self.name!r}: {self._members!r}>'
        else:
            return f'<Element: {self._members!r}>'

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[str]:
        return iter(self._members.keys())

    def keys(self) -> KeysView[str]:
        """Return a view of the valid (casefolded) keys for this element."""
        return self._members.keys()

    def values(self) -> ValuesView[Attribute]:
        """Return a view of the attributes for this element."""
        return self._members.values()

    def __getitem__(self, name: str) -> Attribute:
        return self._members[name.casefold()]

    def __delitem__(self, name: str) -> None:
        """Remove the specified attribute."""
        del self._members[name.casefold()]

    def __setitem__(self, name: str, value: Union[Attribute, ValueT]) -> None:
        """Set a specific value, by deducing the type.

        This means this cannot produce certain types like Time. Use an
        Attribute constructor for that, or set the val_* property of an
        existing attribute.
        """
        if isinstance(value, Attribute):
            value.name = name
            self._members[name.casefold()] = value
            return
        val_type, result = deduce_type(value)
        self._members[name.casefold()] = Attribute(name, val_type, result)

    def clear(self) -> None:
        """Remove all attributes from the element."""
        self._members.clear()

    def pop(self, name: str, default: Union[Attribute, Value, object] = _UNSET) -> Attribute:
        """Remove the specified attribute and return it.

        If not found, an attribute is created from the default if specified,
        or KeyError is raised otherwise.
        """
        key = name.casefold()
        try:
            attr = self._members.pop(key)
        except KeyError:
            if default is _UNSET:
                raise
            if isinstance(default, Attribute):
                return default
            else:
                typ, val = deduce_type(default)
                return Attribute(name, typ, val)
        else:
            return attr

    def popitem(self) -> Tuple[str, Attribute]:
        """Remove and return a (name, attr) pair as a 2-tuple, or raise KeyError."""
        key, attr = self._members.popitem()
        return (attr.name, attr)

    def update(*args: Any, **kwds: Union[Attribute, Value]) -> None:
        """Update from a mapping/iterable, and keyword args.
            If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
            If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
            In either case, this is followed by: for k, v in F.items(): D[k] = v
        """
        if len(args) not in (1, 2):
            raise TypeError(f'Expected 1-2 positional args, not {len(args)}!')
        self: Element = args[0]
        if len(args) == 2:
            other = args[1]
            if isinstance(other, Mapping):
                for key in other:
                    self[key] = other[key]
            elif hasattr(other, "keys"):
                for key in other.keys():
                    self[key] = other[key]
            else:
                for attr in other:
                    if isinstance(attr, Attribute):
                        self._members[attr.name.casefold()] = attr
                    else:
                        key, value = attr
                        self[key] = value
        for key, value in kwds.items():
            self[key] = value

    def setdefault(self, name: str, default: Union[Attribute, Value] = None) -> Attribute:
        """Return the specified attribute name.

        If it does not exist, set it using the default and return that.
        """
        key = name.casefold()
        try:
            return self._members[key]
        except KeyError:
            if not isinstance(default, Attribute):
                typ, val = deduce_type(default)
                default = Attribute(name, typ, val)
            self._members[key] = default
            return default


_NUMBERS = {int, float, bool}
_ANGLES = {Angle, AngleTup}

# Python types to their matching ValueType.
TYPE_TO_VALTYPE: Dict[type, ValueType] = {
    Element: ValueType.ELEMENT,
    int: ValueType.INT,
    float: ValueType.FLOAT,
    bool: ValueType.BOOL,
    str: ValueType.STRING,
    bytes: ValueType.BINARY,
    # Time: ValueType.TIME,
    Color: ValueType.COLOR,
    Vec2: ValueType.VEC2,
    Vec3: ValueType.VEC3,
    Vec4: ValueType.VEC4,
    AngleTup: ValueType.ANGLE,
    Quaternion: ValueType.QUATERNION,
    Matrix: ValueType.MATRIX,
}


def deduce_type(value):
    """Convert Python objects to an appropriate ValueType."""
    if isinstance(value, list):  # Array.
        return deduce_type_array(value)
    else:  # Single value.
        return deduce_type_single(value)


def deduce_type_array(value: list):
    """Convert a Python list to an appropriate ValueType."""
    if len(value) == 0:
        raise TypeError('Cannot deduce type for empty list!')
    types = set(map(type, value))
    if len(types) > 1:
        if types <= _NUMBERS:
            # Allow mixing numerics, casting to the largest subset.
            if float in types:
                return ValueType.FLOAT, list(map(float, value))
            if int in types:
                return ValueType.INTEGER, list(map(int, value))
            if bool in types:
                return ValueType.BOOL, list(map(bool, value))
            raise AssertionError('No numbers?', value)
        elif types == _ANGLES:
            return ValueType.ANGLE, list(map(AngleTup._make, value))
        # Else, fall through and try iterables.
    else:
        [val_actual_type] = types
        if val_actual_type is Matrix:
            return ValueType.MATRIX, [mat.copy() for mat in value]
        if val_actual_type is Angle:
            return ValueType.ANGLE, [AngleTup._make(ang) for ang in value]
        elif val_actual_type is Color:
            return ValueType.COLOR, [
                Color(int(r), int(g), int(b), int(a))
                for r, g, b, a in value
            ]
        try:
            # Match to one of the types directly.
            val_type = TYPE_TO_VALTYPE[val_actual_type]
        except KeyError:
            pass
        else:
            # NamedTuple, ensure they're a float.
            if issubclass(val_actual_type, tuple):
                return val_type, [
                    tuple.__new__(val_actual_type, map(float, val))
                    for val in value
                ]
            else:
                return val_type, value.copy()
    # Deduce each type in the array, check they're the same.
    val_type, first = deduce_type_single(value[0])
    result = [first]
    for val in value[1:]:
        sec_type, sec_val = deduce_type_single(val)
        if sec_type is not val_type:
            raise TypeError(
                'Arrays must have the same types, or be all numeric. '
                f'Got {val_type.name} and {sec_type.name}.'
            )
        result.append(sec_val)
    return val_type, result


def deduce_type_single(value):
    if isinstance(value, Matrix):
        return ValueType.MATRIX, value.copy()
    if isinstance(value, Angle):  # Mutable version of values we use.
        return ValueType.ANGLE, AngleTup._make(value)
    elif isinstance(value, Color):
        [r, g, b, a] = value
        return ValueType.COLOR, Color(int(r), int(g), int(b), int(a))
    try:
        # Match to one of the types directly.
        val_type = TYPE_TO_VALTYPE[type(value)]
    except KeyError:
        # Try iterables.
        pass
    else:
        # NamedTuple, ensure they're a float.
        if isinstance(value, tuple):
            return val_type, tuple.__new__(type(value), map(float, value))
        else:  # No change needed.
            return val_type, value
    try:
        it = iter(value)
    except TypeError:
        # Nope, not an iterable. So not valid.
        raise TypeError(f'Could not deduce value type for {type(value)}.') from None
    # Now determine the length.
    try:
        x = float(next(it))
    except StopIteration:
        raise TypeError(f'Could not deduce vector type for zero-length iterable {type(value)}.') from None
    try:
        y = float(next(it))
    except StopIteration:
        raise TypeError(f'Could not deduce vector type for one-long iterable {type(value)}.') from None
    try:
        z = float(next(it))
    except StopIteration:
        return ValueType.VEC2, Vec2(x, y)
    try:
        w = float(next(it))
    except StopIteration:
        return ValueType.VEC3, Vec3(x, y, z)
    try:
        next(it)
    except StopIteration:
        return ValueType.VEC4, Vec4(x, y, z, w)
    else:
        raise TypeError(f'Could not deduce vector type for 5+ long iterable {type(value)}.') from None


# All the type converter functions.
# Assign to globals, then _get_converters() will find and store these funcs,
# removing them from globals.
# Conversion to/from strings and binary are required for all types.

_conv_string_to_float = float
_conv_string_to_integer = int
_conv_string_to_time = float
_conv_string_to_bool = lambda val: BOOL_LOOKUP[val.casefold()]
_conv_string_to_vec2 = lambda text: Vec2._make(parse_vector(text, 2))
_conv_string_to_vec3 = lambda text: Vec3._make(parse_vector(text, 3))
_conv_string_to_vec4 = lambda text: Vec4._make(parse_vector(text, 4))
_conv_string_to_color = lambda text: Color._make(parse_vector(text, 4))
_conv_string_to_angle = lambda text: AngleTup._make(parse_vector(text, 3))
_conv_string_to_quaternion = lambda text: Quaternion._make(parse_vector(text, 4))

_conv_integer_to_string = str
_conv_integer_to_float = float
_conv_integer_to_time = float
_conv_integer_to_bool = bool
_conv_integer_to_vec2 = lambda n: Vec2(n, n)
_conv_integer_to_vec3 = lambda n: Vec3(n, n, n)
_conv_integer_to_vec4 = lambda n: Vec4(n, n, n, n)


def _conv_integer_to_color(val: int) -> Color:
    val = max(0, min(val, 255))
    return Color(val, val, val, 255)


def _fmt_float(x: float) -> str:
    """Format floats appropriately, with 6 decimal points but no trailing zeros."""
    res = format(x, '.6f').rstrip('0')
    if res.endswith('.'):
        return res[:-1]
    return res

_conv_float_to_string = _fmt_float
_conv_float_to_integer = int
_conv_float_to_bool = bool
_conv_float_to_time = float
_conv_float_to_vec2 = lambda n: Vec2(n, n)
_conv_float_to_vec3 = lambda n: Vec3(n, n, n)
_conv_float_to_vec4 = lambda n: Vec4(n, n, n, n)

_conv_bool_to_integer = int
_conv_bool_to_float = int
_conv_bool_to_string = bool_as_int

_conv_time_to_integer = int
_conv_time_to_float = float
_conv_time_to_bool = lambda t: t > 0
_conv_time_to_string = str

_conv_vec2_to_string = lambda v: f'{_fmt_float(v.x)} {_fmt_float(v.y)}'
_conv_vec2_to_bool = lambda v: bool(v.x or v.y)
_conv_vec2_to_vec3 = lambda v: Vec3(v.x, v.y, 0.0)
_conv_vec2_to_vec4 = lambda v: Vec4(v.x, v.y, 0.0, 0.0)

_conv_vec3_to_string = lambda v: f'{_fmt_float(v.x)} {_fmt_float(v.y)} {_fmt_float(v.z)}'
_conv_vec3_to_bool = lambda v: bool(v.x or v.y or v.z)
_conv_vec3_to_vec2 = lambda v: Vec2(v.x, v.y)
_conv_vec3_to_vec4 = lambda v: Vec4(v.x, v.y, v.z, 0.0)
_conv_vec3_to_angle = lambda v: AngleTup(v.x, v.y, v.z)
_conv_vec3_to_color = lambda v: Color(int(v.x), int(v.y), int(v.z), 255)

_conv_vec4_to_string = lambda v: f'{_fmt_float(v.x)} {_fmt_float(v.y)} {_fmt_float(v.z)} {_fmt_float(v.w)}'
_conv_vec4_to_bool = lambda v: bool(v.x or v.y or v.z or v.w)
_conv_vec4_to_vec3 = lambda v: Vec3(v.x, v.y, v.z)
_conv_vec4_to_vec2 = lambda v: Vec2(v.x, v.y)
_conv_vec4_to_quaternion = lambda v: Quaternion(v.x, v.y, v.z, v.w)
_conv_vec4_to_color = lambda v: Color(int(v.x), int(v.y), int(v.z), int(v.w))

_conv_matrix_to_angle = lambda mat: AngleTup._make(mat.to_angle())

_conv_angle_to_string = lambda a: f'{_fmt_float(a.pitch)} {_fmt_float(a.yaw)} {_fmt_float(a.roll)}'
_conv_angle_to_matrix = lambda ang: Matrix.from_angle(Angle(ang))
_conv_angle_to_vec3 = lambda ang: Vec3(ang.pitch, ang.yaw, ang.roll)

_conv_color_to_string = lambda col: f'{col.r} {col.g} {col.b} {col.a}'
_conv_color_to_vec3 = lambda col: Vec3(col.r, col.g, col.b)
_conv_color_to_vec4 = lambda col: Vec4(col.r, col.g, col.b, col.a)

_conv_quaternion_to_string = lambda quat: f'{_fmt_float(quat.x)} {_fmt_float(quat.y)} {_fmt_float(quat.z)} {_fmt_float(quat.w)}'
_conv_quaternion_to_vec4 = lambda quat: Vec4(quat.x, quat.y, quat.z, quat.w)

# Binary conversions.
_conv_string_to_binary = bytes.fromhex
if sys.version_info >= (3, 8, 0):
    def _conv_binary_to_string(byt: bytes) -> str:
        """Format each byte, seperated by spaces."""
        return byt.hex(' ', 1).upper()
else:
    def _conv_binary_to_string(byt: bytes) -> str:
        """The parameters for bytes.hex aren't available, do it ourselves."""
        return ' '.join([format(x, '02X') for x in byt])


def _binconv_basic(name: str, fmt: str):
    """Converter functions for a type with a single value."""
    shape = Struct(fmt)
    def unpack(byt):
        [val] = shape.unpack(byt)
        return val
    ns = globals()
    ns['_struct_' + name] = shape
    ns[f'_conv_{name}_to_binary'] = shape.pack
    ns[f'_conv_binary_to_{name}'] = unpack


def _binconv_ntup(name: str, fmt: str, Tup: Type[tuple]):
    """Converter functions for a type matching a namedtuple."""
    shape = Struct(fmt)
    ns = globals()
    ns['_struct_' + name] = shape
    ns[f'_conv_{name}_to_binary'] = lambda val: shape.pack(*val)
    ns[f'_conv_binary_to_{name}'] = lambda byt: Tup._make(shape.unpack(byt))

_binconv_basic('integer', '<i')
_binconv_basic('float', '<f')
_binconv_basic('bool', '<?')

_binconv_ntup('color', '<4B', Color)
_binconv_ntup('angle', '<3f', AngleTup)
_binconv_ntup('quaternion', '<4f', Quaternion)

_binconv_ntup('vec2', '<2f', Vec2)
_binconv_ntup('vec3', '<3f', Vec3)
_binconv_ntup('vec4', '<4f', Vec4)

_struct_time = Struct('<i')
def _conv_time_to_binary(tim: Time) -> bytes:
    """Time is written as a fixed point integer."""
    return _struct_time.pack(int(round(tim * 10000.0)))

def _conv_binary_to_time(byt: bytes) -> Time:
    [num] = _struct_time.unpack(byt)
    return Time(num / 10000.0)

_struct_matrix = Struct('<16f')
def _conv_matrix_to_binary(mat):
    """We only set the 3x3 part."""
    return _struct_matrix.pack(
        mat[0, 0], mat[0, 1], mat[0, 2], 0.0,
        mat[1, 0], mat[1, 1], mat[1, 2], 0.0,
        mat[2, 0], mat[2, 1], mat[2, 2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
def _conv_binary_to_matrix(byt):
    """We only set the 3x3 part."""
    data = _struct_matrix.unpack(byt)
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = data[0:3]
    mat[1, 0], mat[1, 1], mat[1, 2] = data[4:7]
    mat[2, 0], mat[2, 1], mat[2, 2] = data[8:11]
    return mat


def _conv_string_to_matrix(text: str) -> Matrix:
    data = parse_vector(text, 16)
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = data[0:3]
    mat[1, 0], mat[1, 1], mat[1, 2] = data[4:7]
    mat[2, 0], mat[2, 1], mat[2, 2] = data[8:11]
    return mat


def _conv_matrix_to_string(mat: Matrix) -> str:
    return (
        f'{mat[0, 0]} {mat[0, 1]} {mat[0, 2]} 0.0\n'
        f'{mat[1, 0]} {mat[1, 1]} {mat[1, 2]} 0.0\n'
        f'{mat[2, 0]} {mat[2, 1]} {mat[2, 2]} 0.0\n'
        '0.0 0.0 0.0 1.0'
    )

# Element ID
_struct_element = Struct('<i')

# Property setter implementations:
_conv_integer = int
_conv_string = str
_conv_binary = bytes
_conv_float = float
_conv_time = float
_conv_bool = bool

def _converter_ntup(typ):
    """Common logic for named-tuple members."""
    def _convert(value) -> typ:
        if isinstance(value, typ):
            return value
        else:
            return typ._make(value)
    return _convert

_conv_color = _converter_ntup(Color)
_conv_vec2 = _converter_ntup(Vec2)
_conv_vec3 = _converter_ntup(Vec3)
_conv_vec4 = _converter_ntup(Vec4)
_conv_quaternion = _converter_ntup(Quaternion)
del _converter_ntup


def _conv_angle(value) -> AngleTup:
    if isinstance(value, AngleTup):
        return value
    elif isinstance(value, Matrix):
        return AngleTup._make(value.to_angle())
    else:
        return AngleTup._make(value)


def _conv_matrix(value) -> Matrix:
    if isinstance(value, AngleTup):
        value = Matrix.from_angle(Angle(*value))
    elif isinstance(value, Angle):
        value = Matrix.from_angle(value)
    elif not isinstance(value, Matrix):
        raise ValueError('Matrix attributes must be set to a matrix.')
    return value.copy()


def _conv_element(value) -> Element:
    if not isinstance(value, Element):
        raise ValueError('Element arrays must contain elements!')
    return value

# Gather up all these functions, add to the dicts.
TYPE_CONVERT, CONVERSIONS, SIZES = _get_converters()
