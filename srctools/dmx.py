"""Handles DataModel eXchange trees, in both binary and text (keyvalues2) format."""
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Type, Generic, Iterable, NewType,
    Dict, Tuple, Callable, IO,
)
import builtins as blt
import struct
import io
import re

from srctools import binformat, bool_as_int, Vec, BOOL_LOOKUP, Matrix, Angle, Vec_tuple as Vec3


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


def _get_converters() -> dict:
    conv = {}
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
                    pass
    return conv


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


class Element(Generic[ValueT], _ValProps):
    """An element in a DMX tree."""
    typ: ValueType
    _value: Value

    def __init__(self, name, typ=ValueType.VOID, val=None):
        """For internal use only."""
        self.name = name
        self.typ = typ
        self._value = val

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
            br'<!--\s*dmx\s+encoding\s+([a-z]+)\s+([0-9]+)\s+'
            br'format\s+([a-z]+)\s+([0-9]+)\s*-->', header,
        )
        if match is None:
            raise ValueError(f'Invalid DMX header {bytes(header)!r}!')
        enc_name, enc_vers_by, fmt_name_by, fmt_vers_by = match.groups()

        enc_vers = int(enc_vers_by.decode('ascii'))
        fmt_name = fmt_name_by.decode('ascii')
        fmt_vers = int(fmt_vers_by.decode('ascii'))
        if enc_name == b'keyvalues2':
            return cls.parse_kv2(io.TextIOWrapper(file), enc_vers), fmt_name, fmt_vers
        elif enc_name == b'binary':
            return cls.parse_bin(file, enc_vers), fmt_name, fmt_vers
        else:
            raise ValueError(f'Unknown DMX encoding {repr(enc_name)[2:-1]}!')

    @classmethod
    def parse_bin(cls, file: IO[bytes], version: int) -> 'Element':
        """Parse the core binary data in a DMX file.

        The <!-- --> format comment line should have already be read.
        """

    @classmethod
    def parse_kv2(cls, file: IO[str], version: int) -> 'Element':
        """Parse a DMX file encoded in KeyValues2.

        The <!-- --> format comment line should have already be read.
        """

    @classmethod
    def int(cls, name, value):
        """Create an element with an integer value."""
        return cls(name, ValueType.INT, value)

    @classmethod
    def float(cls, name: str, value):
        """Create an element with a float value."""
        return cls(name, ValueType.FLOAT, value)

    @classmethod
    def bool(cls, name, value):
        """Create an element with a boolean value."""
        return cls(name, ValueType.BOOL, value)

    @classmethod
    def string(cls, name, value):
        """Create an element with a string value."""
        return cls(name, ValueType.STRING, value)

    @classmethod
    def binary(cls, name: str, value):
        """Create an element with binary data."""
        return cls(name, ValueType.BINARY, value)

    @classmethod
    def vec2(cls, name, x=0.0, y=0.0):
        """Create an element with a 2D vector."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
        return cls(name, ValueType.VEC2, Vec2(x, y))

    @classmethod
    def vec3(cls, name, x=0.0, y=0.0, z=0.0):
        """Create an element with a 3D vector."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
        return cls(name, ValueType.VEC3, Vec3(x, y, z))

    @classmethod
    def vec4(cls, name, x=0.0, y=0.0, z=0.0, w=0.0):
        """Create an element with a 4D vector."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
            w = float(next(it, w))
        return cls(name, ValueType.VEC4, Vec4(x, y, z, w))

    @classmethod
    def color(cls, name, r=0.0, g=0.0, b=0.0):
        """Create an element with a color."""
        if not isinstance(r, (int, float)):
            it = iter(r)
            r = float(next(it, 0.0))
            g = float(next(it, g))
            b = float(next(it, b))
        return cls(name, ValueType.COLOR, Color(r, g, b))

    @classmethod
    def angle(cls, name, pitch=0.0, yaw=0.0, roll=0.0):
        """Create an element with an Euler angle."""
        if not isinstance(pitch, (int, float)):
            it = iter(pitch)
            pitch = float(next(it, 0.0))
            yaw = float(next(it, yaw))
            roll = float(next(it, roll))
        return cls(name, ValueType.ANGLE, AngleTup(pitch, yaw, roll))

    @classmethod
    def quaternion(
        cls,
        name: str,
        x: Union[blt.float, Iterable[blt.float]] = 0.0,
        y: blt.float = 0.0,
        z: blt.float = 0.0,
        w: blt.float = 0.0,
    ) -> 'Element[Quaternion]':
        """Create an element with a quaternion rotation."""
        if not isinstance(x, (int, float)):
            it = iter(x)
            x = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
            w = float(next(it, w))
        return cls(name, ValueType.QUATERNION, Quaternion(x, y, z, w))

    def _read_val(self, newtype: ValueType) -> Value:
        """Convert to the desired type."""
        if isinstance(self._value, list):
            raise ValueError('Cannot read value of array elements!')
        try:
            func = _CONVERSIONS[self.typ, newtype]
        except KeyError:
            raise ValueError(f'Cannot convert ({self._value}) to {newtype} type!')
        return func(self._value)

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        self.typ = newtype
        self._value = value

    def __repr__(self) -> str:
        return f'<{self.typ.name} Element {self.name!r}: {self._value!r}>'

# All the type converter functions.
# Assign to globals, then _get_converters() will find and store these funcs,
# removing them from globals.
# Conversion to/from strings and binary are required for all types.

_conv_string_to_float = float
_conv_string_to_integer = int
_conv_string_to_time = float
_conv_string_to_bool = lambda val: BOOL_LOOKUP[val.casefold()]

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

_conv_vec4_to_string = lambda v: f'{v.x:g} {v.y:g} {v.z:g} {v.w:g}'
_conv_vec4_to_bool = lambda v: bool(v.x or v.y or v.z or v.w)
_conv_vec4_to_vec3 = lambda v: Vec3(v.x, v.y, v.z)
_conv_vec4_to_vec2 = lambda v: Vec2(v.x, v.y)

_conv_matrix_to_string = str
_conv_matrix_to_angle = lambda mat: AngleTup._make(mat.to_angle())

_conv_angle_to_string = lambda a: f'{a.pitch:g} {a.yaw:g} {a.roll:g}'
_conv_angle_to_matrix = lambda ang: Matrix.from_angle(Angle(ang))
_conv_angle_to_vec3 = lambda ang: Vec3(ang.pitch, ang.yaw, ang.roll)

# Binary conversions.
_conv_string_to_binary = lambda text: text.encode('ascii') + '\0'
_conv_binary_to_string = lambda binary: binary.decode('ascii')

_conv_integer_to_binary = lambda num: struct.pack('<i', num)
_conv_binary_to_integer = lambda byt: struct.unpack('<i', byt)[0]

_CONVERSIONS = _get_converters()
