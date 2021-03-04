"""Handles DataModel eXchange trees, in both binary and text (keyvalues2) format."""
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Generic, Iterable, NewType,
    Dict, Tuple, Callable, IO, List,
)
import builtins as blt
from struct import Struct
import io
import re
from uuid import UUID, uuid4 as get_uuid

from srctools import binformat, bool_as_int, Vec, BOOL_LOOKUP, Matrix, Angle, Vec_tuple as Vec3
from srctools.tokenizer import Py_Tokenizer as Tokenizer, Token


class ValueType(Enum):
    """The type of value an element has."""
    ELEMENT = 'element'  # Another attribute
    INTEGER = INT = 'int'
    FLOAT = 'float'
    BOOL = 'bool'
    STRING = STR = 'string'
    BINARY = BIN = VOID = 'void'  # IE "void *", binary blob.
    TIME = 'time'  # Seconds
    COLOR = COLOUR = 'color'
    VEC2 = 'vector2'
    VEC3 = 'vector3'
    VEC4 = 'vector4'
    ANGLE = 'qangle'
    QUATERNION = 'quaternion'
    MATRIX = 'vmatrix'


class Vec2(NamedTuple):
    """A 2-dimensional vector."""
    x: float
    y: float


class Vec4(NamedTuple):
    """A 4-dimensional vector."""
    x: float
    y: float
    z: float
    w: float


class Quaternion(NamedTuple):
    """A quaternion used to represent rotations."""
    x: float
    y: float
    z: float
    w: float


class Color(NamedTuple):
    """An RGB color."""
    r: float
    g: float
    b: float


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
    NewType,
    Vec2, Vec3,
    Vec4,
    AngleTup,
    Quaternion,
    Matrix,
    'Element',
]

ValueT = TypeVar(
    'ValueT',
    int, float, bool, str, bytes,
    Color,
    Vec2, Vec3, Vec4,
    Angle,
    Quaternion,
    Matrix,
    'Element',
)

# [from, to] -> conversion.
# Implementation at the end of the file.
_CONVERSIONS: Dict[Tuple[ValueType, ValueType], Callable[[Value], Value]]
# And type -> size, excluding str/bytes.
SIZES: Dict[ValueType, int]


def parse_vector(text: str, count: int) -> List[float]:
    """Parse a space-delimited vector."""
    parts = text.split()
    if len(parts) != count:
        raise ValueError(f'{text!r} is not a {count}-dimensional vector!')
    return list(map(float, parts))


def _get_converters() -> Tuple[dict, dict]:
    conv = {}
    sizes = {}
    ns = globals()

    def unchanged(x):
        """No change means no conversion needed."""
        return x

    for from_typ in ValueType:
        for to_typ in ValueType:
            if from_typ is to_typ:
                conv[from_typ, to_typ] = unchanged
            else:
                func = f'_conv_{from_typ.name.casefold()}_to_{to_typ.name.casefold()}'
                try:
                    conv[from_typ, to_typ] = ns.pop(func)
                except KeyError:
                    if (from_typ is ValueType.STRING or to_typ is ValueType.STRING) and from_typ is not ValueType.ELEMENT and to_typ is not ValueType.ELEMENT:
                        raise ValueError(func + ' must exist!')
        # Special cases, variable size.
        if from_typ is not ValueType.STRING and from_typ is not ValueType.BINARY:
            sizes[from_typ] = ns['_struct_' + from_typ.name.casefold()].size

    return conv, sizes


def _make_val_prop(val_type: ValueType, typ: type) -> property:
    """Build the properties for each type."""

    def setter(self, value):
        self._write_val(val_type, value)

    def getter(self):
        return self._read_val(val_type)
    if val_type.name[0].casefold() in 'aeiou':
        desc = f' the value as an {val_type.name.lower()}.'
    else:
        desc = f' the value as a {val_type.name.lower()}.'
    getter.__doc__ = 'Return' + desc
    setter.__doc__ = 'Convert' + desc
    getter.__annotations__['return'] = typ
    setter.__annotations__['value'] = typ
    return property(
        fget=getter,
        fset=setter,
        doc=f'Access' + desc,
    )


class _ValProps:
    """Properties which read/write as the various kinds of value types."""
    def _read_val(self, newtype: ValueType) -> Value:
        """Convert to the desired type."""
        raise NotImplementedError

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        """Set to the desired type."""
        raise NotImplementedError

    def set_val_void(self) -> None:
        """Set the value to void (no value).

        Unlike other types this is a method since no value needs to be read.
        """
        self._write_val(ValueType.VOID, None)

    val_int = _make_val_prop(ValueType.INT, int)
    val_str = val_string = _make_val_prop(ValueType.STRING, str)
    val_bin = val_binary = _make_val_prop(ValueType.BINARY, bytes)
    val_float = _make_val_prop(ValueType.FLOAT, float)
    val_bool = _make_val_prop(ValueType.BOOL, bool)
    val_colour = val_color = _make_val_prop(ValueType.COLOR, Color)
    val_vec2 = _make_val_prop(ValueType.VEC2, Vec2)
    val_vec3 = _make_val_prop(ValueType.VEC3, Vec3)
    val_vec4 = _make_val_prop(ValueType.VEC4, Vec4)
    val_quat = val_quaternion = _make_val_prop(ValueType.QUATERNION, Quaternion)
    val_ang = val_angle = _make_val_prop(ValueType.ANGLE, AngleTup)
    val_mat = val_matrix = _make_val_prop(ValueType.MATRIX, Matrix)

del _make_val_prop


class ElemMember(_ValProps):
    """A proxy for individual indexes/keys, allowing having .val attributes."""
    def __init__(self, owner, index):
        self.owner = owner
        self.index = index

    def _read_val(self, newtype: ValueType) -> Value:
        if isinstance(self.owner._value, (list, dict)):
            value = self.owner._value[self.index]
        else:
            value = self.owner._value
        try:
            func = _CONVERSIONS[self.owner._val_typ, newtype]
        except KeyError:
            raise ValueError(f'Cannot convert ({value}) to {newtype} type!')
        return func(value)

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        if newtype != self.owner._val_typ:
            raise ValueError('Cannot change type of array.')
        if isinstance(self.owner._value, (list, dict)):
            self.owner._value[self.index] = value
        else:
            self.owner._value = value


class Element(Generic[ValueT], _ValProps):
    """An element in a DMX tree."""
    type: str
    name: str
    uuid: UUID
    _val_typ: Union[ValueType, str]
    _value: Union[Value, list, dict]

    def __init__(self, el_type, val_typ, val, uuid: UUID=None, name=''):
        """For internal use only."""
        self.type = el_type
        self.name = name
        self._val_typ = val_typ
        self._value = val
        if uuid is None:
            self.uuid = get_uuid()
        else:
            self.uuid = uuid

    @classmethod
    def parse(cls, file: IO[bytes]) -> Tuple['Element', str, int]:
        """Parse a DMX file encoded in binary or KV2 (text).

        The return value is the tree, format name and version.
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
            br'<!--\s*dmx\s+encoding\s+([a-z0-9]+)\s+([0-9]+)\s+'
            br'format\s+([a-z0-9]+)\s+([0-9]+)\s*-->', header,
        )
        if match is not None:
            enc_name, enc_vers_by, fmt_name_by, fmt_vers_by = match.groups()

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
            enc_vers = 0
            fmt_name = ''
            fmt_vers = 0

        if enc_name == b'keyvalues2':
            result = cls.parse_kv2(io.TextIOWrapper(file), enc_vers)
        elif enc_name == b'binary':
            result = cls.parse_bin(file, enc_vers)
        else:
            raise ValueError(f'Unknown DMX encoding {repr(enc_name)[2:-1]}!')

        return result, fmt_name, fmt_vers

    @classmethod
    def parse_bin(cls, file: IO[bytes], version: int) -> 'Element':
        """Parse the core binary data in a DMX file.

        The <!-- --> format comment line should have already be read.
        """

    @classmethod
    def parse_kv2(cls, file: IO[str], version: int) -> List['Element']:
        """Parse a DMX file encoded in KeyValues2.

        The <!-- --> format comment line should have already be read.
        """
        # For fixup after it's fully parsed.
        id_to_elem: Dict[UUID, Element] = {}

        elements = []

        tok = Tokenizer(file)
        for token, tok_value in tok:
            if token is Token.STRING:
                elem_name = tok_value
            elif token is Token.NEWLINE:
                continue
            else:
                raise tok.error(token)
            elements.append(cls._parse_kv2_element(tok, id_to_elem, elem_name))

        # Now assign UUIDs.
        for elem in id_to_elem.values():
            if elem._val_typ is not ValueType.ELEMENT:
                continue
            if isinstance(elem._value, list):
                iterator = enumerate(elem._value)
            elif isinstance(elem._value, dict):
                iterator = elem._value.items()
            elif isinstance(elem._value, UUID):
                try:
                    elem._value = id_to_elem[elem._value]
                except KeyError:
                    raise tok.error('UUID {} not found!', elem._value)
                continue
            else:
                continue
            for key, value in iterator:
                if isinstance(value, UUID):
                    try:
                        elem._value[key] = id_to_elem[value]
                    except KeyError:
                        raise tok.error('UUID {} not found!', value)
        return elements

    @classmethod
    def _parse_kv2_element(cls, tok, id_to_elem, name):
        """Parse a compound element."""
        elem: Element = cls(name, ValueType.ELEMENT, {})
        for attr_name in tok.block(name):
            orig_typ_name = tok.expect(Token.STRING)
            typ_name = orig_typ_name.casefold()

            # The UUID is a special element name/type combo.
            if attr_name.casefold() == 'id':
                if typ_name != 'elementid':
                    raise tok.error(
                        'Element ID attribute must be '
                        '"elementid" type, not "{}"!',
                        typ_name
                    )
                uuid_str = tok.expect(Token.STRING)
                try:
                    elem.uuid = UUID(uuid_str)
                except ValueError:
                    raise tok.error('Invalid UUID "{}"!', uuid_str)
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
                elem._value[attr_name] = cls._parse_kv2_element(tok, id_to_elem, orig_typ_name)
                continue
            if is_array:
                array = []
                attr = Element(attr_name, attr_type, array)
                id_to_elem[attr.uuid] = attr
                tok.expect(Token.BRACK_OPEN)
                for tok_typ, tok_value in tok:
                    if tok_typ is Token.BRACK_CLOSE:
                        break
                    elif tok_typ is Token.STRING:
                        if attr_type is ValueType.ELEMENT:
                            if tok_value == 'element':
                                # UUID reference.
                                uuid_str = tok.expect(Token.STRING)
                                try:
                                    array.append(UUID(uuid_str))
                                except ValueError:
                                    raise tok.error('Invalid UUID "{}"!', uuid_str)
                            else:
                                # Inline compound
                                array.append(cls._parse_kv2_element(tok, id_to_elem, tok_value))
                        else:
                            # Other value
                            try:
                                array.append(_CONVERSIONS[ValueType.STRING, attr_type](tok_value))
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
                # Put the UUID in instead, we'll fix it up later.
                uuid_str = tok.expect(Token.STRING)
                try:
                    attr = UUID(uuid_str)
                except ValueError:
                    raise tok.error('Invalid UUID "{}"!', uuid_str)
            else:
                # Single element.
                unparsed = tok.expect(Token.STRING)
                value = _CONVERSIONS[ValueType.STRING, attr_type](unparsed)
                attr = Element(attr_name, attr_type, value)
            elem._value[attr_name] = attr

        id_to_elem[elem.uuid] = elem
        return elem

    @classmethod
    def int(cls, el_type, value):
        """Create an element with an integer value."""
        return cls(el_type, ValueType.INT, value)

    @classmethod
    def float(cls, el_type: str, value):
        """Create an element with a float value."""
        return cls(el_type, ValueType.FLOAT, value)

    @classmethod
    def bool(cls, el_type, value):
        """Create an element with a boolean value."""
        return cls(el_type, ValueType.BOOL, value)

    @classmethod
    def string(cls, el_type, value):
        """Create an element with a string value."""
        return cls(el_type, ValueType.STRING, value)

    @classmethod
    def binary(cls, el_type: str, value, name=''):
        """Create an element with binary data."""
        return cls(el_type, ValueType.BINARY, value, None, name)

    @classmethod
    def vec2(cls, el_type, x=0.0, y=0.0, name=''):
        """Create an element with a 2D vector."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
        return cls(el_type, ValueType.VEC2, Vec2(x, y), None, name)

    @classmethod
    def vec3(cls, el_type, x=0.0, y=0.0, z=0.0, name=''):
        """Create an element with a 3D vector."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
        return cls(el_type, ValueType.VEC3, Vec3(x, y, z), None, name)

    @classmethod
    def vec4(cls, el_type, x=0.0, y=0.0, z=0.0, w=0.0, name=''):
        """Create an element with a 4D vector."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
            w = float(next(it, w))
        return cls(el_type, ValueType.VEC4, Vec4(x, y, z, w), None, name)

    @classmethod
    def color(cls, el_type, r=0.0, g=0.0, b=0.0, name=''):
        """Create an element with a color."""
        if not isinstance(r, (int, float)):
            it = iter(r)
            r = float(next(it, 0.0))
            g = float(next(it, g))
            b = float(next(it, b))
        return cls(el_type, ValueType.COLOR, Color(r, g, b), None, name)

    @classmethod
    def angle(cls, el_type, pitch=0.0, yaw=0.0, roll=0.0, name=''):
        """Create an element with an Euler angle."""
        if not isinstance(pitch, (int, float)):
            it = iter(pitch)
            pitch = float(next(it, 0.0))
            yaw = float(next(it, yaw))
            roll = float(next(it, roll))
        return cls(el_type, ValueType.ANGLE, AngleTup(pitch, yaw, roll), None, name)

    @classmethod
    def quaternion(
        cls,
        el_type: str,
        x: Union[blt.float, Iterable[blt.float]] = 0.0,
        y: blt.float = 0.0,
        z: blt.float = 0.0,
        w: blt.float = 0.0,
        name='',
    ) -> 'Element[Quaternion]':
        """Create an element with a quaternion rotation."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
            w = float(next(it, w))
        return cls(el_type, ValueType.QUATERNION, Quaternion(x, y, z, w), None, name)

    def _read_val(self, newtype: ValueType) -> Value:
        """Convert to the desired type."""
        if isinstance(self._value, list):
            raise ValueError('Cannot read value of array elements!')
        if isinstance(self._value, dict):
            raise ValueError('Cannot read value of compound elements!')
        try:
            func = _CONVERSIONS[self._val_typ, newtype]
        except KeyError:
            raise ValueError(f'Cannot convert ({self._value}) to {newtype} type!')
        return func(self._value)

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        self._val_typ = newtype
        self._value = value

    def __repr__(self) -> str:
        return f'<Element {self.name!r}: {self._value!r}>'

    def __getitem__(self, item) -> ElemMember:
        """Read values in an array element."""
        if not isinstance(self._value, (list, dict)):
            raise ValueError('Cannot index singular elements.')
        _ = self._value[item]  # Raise IndexError/KeyError if not present.
        return ElemMember(self, item)

    def __setitem__(self, key, value):
        """Set a specific array element to a value."""


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
_conv_string_to_color = lambda text: Color._make(parse_vector(text, 3))
_conv_string_to_angle = lambda text: AngleTup._make(parse_vector(text, 3))
_conv_string_to_quaternion = lambda text: Quaternion._make(parse_vector(text, 4))

_conv_integer_to_string = str
_conv_integer_to_float = float
_conv_integer_to_time = float
_conv_integer_to_bool = bool
_conv_integer_to_vec2 = lambda n: Vec2(n, n)
_conv_integer_to_vec3 = lambda n: Vec3(n, n, n)
_conv_integer_to_vec4 = lambda n: Vec4(n, n, n, n)

_conv_float_to_string = '{:g}'.format
_conv_float_to_integer = int
_conv_float_to_bool = bool
_conv_float_to_time = float
_conv_float_to_vec2 = lambda n: Vec2(n, n)
_conv_float_to_vec3 = lambda n: Vec3(n, n, n)
_conv_float_to_vec4 = lambda n: Vec4(n, n, n, n)

def _conv_float_to_color(val: float) -> Color:
    val = max(0, min(val, 255))
    return Color(val, val, val)

_conv_bool_to_integer = int
_conv_bool_to_float = int
_conv_bool_to_string = bool_as_int

_conv_time_to_integer = int
_conv_time_to_float = float
_conv_time_to_bool = lambda t: t > 0
_conv_time_to_string = str

_conv_vec2_to_string = lambda v: f'{v.x:g} {v.y:g}'
_conv_vec2_to_bool = lambda v: bool(v.x or v.y)
_conv_vec2_to_vec3 = lambda v: Vec3(v.x, v.y, 0.0)
_conv_vec2_to_vec4 = lambda v: Vec4(v.x, v.y, 0.0, 0.0)

_conv_vec3_to_string = lambda v: f'{v.x:g} {v.y:g} {v.z:g}'
_conv_vec3_to_bool = lambda v: bool(v.x or v.y or v.z)
_conv_vec3_to_vec2 = lambda v: Vec2(v.x, v.y)
_conv_vec3_to_vec4 = lambda v: Vec4(v.x, v.y, v.z, 0.0)
_conv_vec3_to_angle = lambda v: AngleTup(v.x, v.y, v.z)
_conv_vec3_to_color = lambda v: Color(v.x, v.y, v.z)

_conv_vec4_to_string = lambda v: f'{v.x:g} {v.y:g} {v.z:g} {v.w:g}'
_conv_vec4_to_bool = lambda v: bool(v.x or v.y or v.z or v.w)
_conv_vec4_to_vec3 = lambda v: Vec3(v.x, v.y, v.z)
_conv_vec4_to_vec2 = lambda v: Vec2(v.x, v.y)
_conv_vec4_to_quaternion = lambda v: Quaternion(v.x, v.y, v.z, v.w)

_conv_matrix_to_string = str
_conv_matrix_to_angle = lambda mat: AngleTup._make(mat.to_angle())

_conv_angle_to_string = lambda a: f'{a.pitch:g} {a.yaw:g} {a.roll:g}'
_conv_angle_to_matrix = lambda ang: Matrix.from_angle(Angle(ang))
_conv_angle_to_vec3 = lambda ang: Vec3(ang.pitch, ang.yaw, ang.roll)

_conv_color_to_string = lambda col: f'{col.r:g} {col.g:g} {col.b:g}'
_conv_color_to_vec3 = lambda col: Vec3(col.r, col.g, col.b)

_conv_quaternion_to_string = lambda quat: f'{quat.x:g} {quat.y:g} {quat.z:g} {quat.w:g}'
_conv_quaternion_to_vec4 = lambda quat: Vec4(quat.x, quat.y, quat.z, quat.w)

# Binary conversions.
_conv_string_to_binary = lambda text: text.encode('ascii') + '\0'
_conv_binary_to_string = lambda binary: binary.decode('ascii')


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


def _binconv_ntup(name: str, fmt: str, Tup):
    """Converter functions for a type matching a namedtuple."""
    shape = Struct(fmt)
    ns = globals()
    ns['_struct_' + name] = shape
    ns[f'_conv_{name}_to_binary'] = lambda val: shape.pack(*val)
    ns[f'_conv_binary_to_{name}'] = lambda byt: Tup._make(shape.unpack(byt))

_binconv_basic('integer', '<i')
_binconv_basic('float', '<f')
_binconv_basic('time', '<f')
_binconv_basic('bool', '<?')

_binconv_ntup('color', '<3f', Color)
_binconv_ntup('angle', '<3f', AngleTup)
_binconv_ntup('quaternion', '<4f', Quaternion)

_binconv_ntup('vec2', '<2f', Vec2)
_binconv_ntup('vec3', '<3f', Vec3)
_binconv_ntup('vec4', '<4f', Vec4)

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

def _conv_string_to_matrix(text):
    data = parse_vector(text, 16)
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = data[0:3]
    mat[1, 0], mat[1, 1], mat[1, 2] = data[4:7]
    mat[2, 0], mat[2, 1], mat[2, 2] = data[8:11]
    return mat

def _conv_matrix_to_string(mat):
    return (
        f'{mat[0, 0]:g} {mat[0, 1]:g} {mat[0, 2]:g} 0 '
        f'{mat[1, 0]:g} {mat[1, 1]:g} {mat[1, 2]:g} 0 '
        f'{mat[2, 0]:g} {mat[2, 1]:g} {mat[2, 2]:g} 0 '
        '0 0 0 1'
    )

# Element ID
_struct_element = Struct('<i')

_CONVERSIONS, SIZES = _get_converters()
