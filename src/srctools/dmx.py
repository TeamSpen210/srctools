"""Handles DataModel eXchange trees, in both binary and text (keyvalues2) format.

As an extension, optionally all strings may become full UTF-8, marked by a new
set of 'unicode_XXX' encoding formats.

Special 'stub' elements are possible, which represent elements not present in the file. 
These are represented by StubElement instances. Additionally, NULL elements are possible.

To parse a DMX file, it must be opened in binary mode (the kind will be detected automatically)::

    with open("path/to/file.dmx", "rb") as f:
        dmx, format_name, format_version = Element.parse(f)

The format name/version are stored in the header, allowing indicating the kind of data stored in 
the file.
"""
from typing import (
    IO, TYPE_CHECKING, Any, Callable, Dict, Generic, Iterable, Iterator, KeysView, List,
    Mapping, MutableMapping, NamedTuple, Optional, Sequence, Set, Tuple, Type, TypeVar, Union,
    ValuesView, cast,
)
from typing_extensions import Final, Literal, TypeAlias, overload
from enum import Enum
from struct import Struct, error as StructError, pack
from uuid import UUID, uuid4 as get_uuid
import builtins
import collections
import copy
import io
import re
import sys
import warnings

import attrs

from srctools import BOOL_LOOKUP, EmptyMapping, binformat, bool_as_int
from srctools.keyvalues import Keyvalues
from srctools.math import Angle, FrozenAngle, FrozenMatrix, FrozenVec, Matrix, Vec
from srctools.tokenizer import Token, Tokenizer


__all__ = [
    'ValueType', 'NULL',
    'STUB', 'Vec3', 'AngleTup',  # Deprecated.
    'Vec2', 'Vec4', 'Quaternion', 'Color', 'Time',
    'Attribute', 'Element', 'StubElement',
]


class ValueType(Enum):
    """The type of value an element attribute has."""
    ELEMENT = 'element'
    """A child :py:class:`Element`. This may additionally be :py:data:`NULL` or a :py:class:`StubElement`, representing data stored elsewhere."""
    INTEGER = INT = 'int'
    """A 32-bit signed :external:py:class:`int` value. In text format any value can be parsed/exported."""
    FLOAT = 'float'
    """A single-precision :external:py:class:`float` point value. In text format the full Python double-precision value can be parsed/exported."""
    BOOL = 'bool'
    """A :external:py:class:`bool` true/false value."""
    STRING = STR = 'string'
    """A general text string, though it should probably be kept to ASCII/CP-1252 to ensure Valve's parsers produce the correct result. See the ``unicode`` option in :py:meth:`Element.parse()`, :py:meth:`Element.export_kv2()` and :py:meth:`~Element.export_bin()`."""
    BINARY = BIN = VOID = 'binary'
    """A block of raw :external:py:class:`bytes` data. Any buffer object can be converted to this."""
    TIME = 'time'
    """Time since the begining of a level, in seconds. This is represented by the :py:class:`Time` type."""
    COLOR = COLOUR = 'color'
    """An RGBA 8-bit colour. This can be converted from a 3 or 4-tuple, or a space separated string. In either case alpha defaults to ``255`` if not specified."""
    VEC2 = 'vector2'
    """A generic XY vector, represented by the :py:class:`Vec2` namedtuple. This can be converted from an iterable of 4 floats."""
    VEC3 = 'vector3'
    """A generic XYZ vector, represented by :py:class:`srctools.math.FrozenVec` classs. This can be converted from an iterable of 3 floats."""
    VEC4 = 'vector4'
    """A generic XYZW vector, represented by the :py:class:`Vec4` namedtuple. This can be converted from an iterable of 4 floats."""
    ANGLE = 'qangle'
    """An Euler angle, represented by :py:class:`srctools.math.FrozenAngle`. This can be converted from an iterable of 3 floats, :py:class:`~srctools.math.Angle`,  :py:class:`~srctools.math.FrozenMatrix` or :py:class:`~srctools.math.Matrix`."""
    QUATERNION = 'quaternion'
    """A rotational quaternion, represented by the :py:class:`Quaternion` namedtuple. This can be converted from a 4-element iterable."""
    MATRIX = 'vmatrix'
    """A translation :py:class:`~srctools.math.FrozenMatrix`. This can be converted from a :py:class:`~srctools.math.Angle`,  :py:class:`~srctools.math.FrozenAngle` or :py:class:`~srctools.math.Matrix`."""


class _StubType(str, Enum):
    """Kind of StubElement."""
    STUB = 'DMEStubElement'
    NULL = 'DMENullElement'


# type -> enum index.
VAL_TYPE_TO_IND: Final = {
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
ARRAY_OFFSET: Final = 14
IND_TO_VALTYPE: Final = {
    ind: val_type
    for val_type, ind in VAL_TYPE_TO_IND.items()
}
# For parsing, set this initially to check one is set.
_UNSET_UUID: Final = get_uuid()
_UNSET: Any = object()  # Argument sentinel
# Deprecated.
STUB = _StubType.STUB  #: :deprecated: Use :py:func:`StubElement.stub()`.
T = TypeVar('T')


class Vec2(NamedTuple):
    """A 2-dimensional vector."""
    x: float
    y: float

    def __repr__(self) -> str:
        return f'({self[0]:.6g} {self[1]:.6g})'


class Vec4(NamedTuple):
    """A 4-dimensional vector."""
    x: float
    y: float
    z: float
    w: float

    def __str__(self) -> str:
        return f'({self[0]:.6g} {self[1]:.6g} {self[2]:.6g} {self[3]:.6g})'


class Quaternion(NamedTuple):
    """A quaternion used to represent rotations."""
    x: float
    y: float
    z: float
    w: float
    # def __repr__(self) -> str:
    #     return f'({self[0]} {self[1]:.6g} {self[2]:.6g} {self[3]:.6g})'


def _clamp_color(x: int) -> int:
    """Clamp colors to 0-255."""
    return max(0, min(255, round(x)))


@attrs.frozen
class Color:
    """An RGB color."""

    r: int = attrs.field(converter=_clamp_color)
    g: int = attrs.field(converter=_clamp_color)
    b: int = attrs.field(converter=_clamp_color)
    a: int = attrs.field(converter=_clamp_color, default=255)

    def __iter__(self) -> Iterator[int]:
        yield self.r
        yield self.g
        yield self.b
        yield self.a

    def __str__(self) -> str:
        return f'{self.r} {self.g} {self.b} {self.a}'


@attrs.frozen
class Time:
    """A relative timestamp."""
    value: float = 0.0

    def __add__(self, other: float) -> 'Time':
        """Only floats can be added, times cannot."""
        if isinstance(other, (int, float)):
            return Time(self.value + other)
        else:
            return NotImplemented

    def __sub__(self, other: 'Time') -> float:
        if isinstance(other, Time):
            return self.value - other.value
        else:
            return NotImplemented

    def __float__(self) -> float:
        """This can be coerced into a float."""
        return self.value


AngleTup = FrozenAngle  #: :deprecated: Use :py:class:`srctools.math.FrozenAngle`.
Vec3 = FrozenVec  #: :deprecated: Use :py:class:`srctools.math.FrozenVec`.


_Element: TypeAlias = 'Element'  # Forward ref.
Value: TypeAlias = Union[
    int, float, bool, str, bytes,
    Color, Time,
    Vec2, FrozenVec, Vec4,
    FrozenAngle,
    Quaternion,
    FrozenMatrix,
    _Element,
]
ValueList = Union[
    List[int], List[float], List[bool], List[str], List[bytes],
    List[Color], List[Time],
    List[Vec2], List[FrozenVec], List[Vec4],
    List[FrozenAngle],
    List[Quaternion],
    List[FrozenMatrix],
    List[_Element],
]
# Additional values we convert to valid types.
ConvValue: TypeAlias = Union[Value, Vec, Matrix, Angle]

ValueT = TypeVar(
    'ValueT',
    int, float, bool, str, bytes,
    Color, Time,
    Vec2, FrozenVec, Vec4,
    FrozenAngle,
    Quaternion,
    FrozenMatrix,
    _Element,
)

# [from, to] -> conversion.
# Implementation at the end of the file.
# Unfortunately we can't show the ValueType -> Value match to the type checker.
ConvertMap: TypeAlias = Dict[Tuple[ValueType, ValueType], Callable[[Value], Value]]
TYPE_CONVERT: ConvertMap

# Take valid types, convert to the value.
ConversionMap: TypeAlias = Dict[ValueType, Callable[[object], Any]]
CONVERSIONS: ConversionMap

# And type -> size, excluding str/bytes.
SizesMap: TypeAlias = Dict[ValueType, int]
SIZES: SizesMap

# Name used for keyvalues1 properties.
NAME_KV1: Final = 'DmElement'
# Additional name, to handle blocks with mixed properties or duplicate names.
NAME_KV1_LEAF: Final = 'DmElementLeaf'
NAME_KV1_ROOT: Final = 'DmElementRoot'


def parse_vector(text: str, count: int) -> List[float]:
    """Parse a space-delimited vector."""
    parts = text.split()
    if len(parts) != count:
        raise ValueError(f'{text!r} is not a {count}-dimensional vector!')
    return list(map(float, parts))


def _get_converters() -> Tuple[ConvertMap, ConversionMap, SizesMap]:
    """Build the conversions maps."""
    type_conv: ConvertMap = {}
    convert: ConversionMap = {}
    sizes: SizesMap = {}
    ns = globals()

    def unchanged(x: T) -> T:
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


def _make_val_prop(val_type: ValueType, typ: Type[Value]) -> property:
    """Build the properties for each type."""

    def setter(self: '_ValProps', value: ValueT) -> None:
        self._write_val(val_type, value)

    def getter(self: '_ValProps') -> ValueT:
        return self._read_val(val_type)  # type: ignore

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


def _make_iter(val_type: ValueType, typ: Type[ValueT]) -> Callable[['Attribute[ValueT]'], Iterator[ValueT]]:
    """Build an iterator for the given value type."""
    def iterator(self: 'Attribute[ValueT]') -> Iterator[ValueT]:
        return self._iter_array(val_type)  # type: ignore

    iterator.__doc__ = f'Iterate over {val_type.name.lower()} values.'
    iterator.__annotations__['return'] = Iterator[typ]  # type: ignore
    return iterator


class _ValProps:
    """Properties which read/write as the various kinds of value types."""

    def _read_val(self, newtype: ValueType) -> Value:
        """Convert to the desired type."""
        raise NotImplementedError

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        """Set to the desired type."""
        raise NotImplementedError

    # Treat these as properties
    if TYPE_CHECKING:
        @property
        def val_int(self) -> int: return 0
        @val_int.setter
        def val_int(self, value: Union[int, float]) -> None: ...

        val_float: float

        @property
        def val_time(self) -> Time: return Time(0.0)
        @val_time.setter
        def val_time(self, value: Union[float, Time]) -> None: ...

        @property
        def val_bool(self) -> bool: return False
        @val_bool.setter
        def val_bool(self, value: object) -> None: ...

        @property
        def val_vec2(self) -> Vec2: return Vec2(0.0, 0.0)
        @val_vec2.setter
        def val_vec2(self, value: Union[Vec2, Tuple[float, float], Iterable[float]]) -> None: ...

        @property
        def val_vec3(self) -> FrozenVec: return FrozenVec()
        @val_vec3.setter
        def val_vec3(self, value: Union[Vec, FrozenVec, Tuple[float, float, float], Iterable[float]]) -> None: ...

        @property
        def val_vec4(self) -> Vec4: return Vec4(0.0, 0.0, 0.0, 1.0)
        @val_vec4.setter
        def val_vec4(self, value: Union[Vec4, Tuple[float, float, float, float], Iterable[float]]) -> None: ...

        @property
        def val_color(self) -> Color: return Color(0, 0, 0)
        @val_color.setter
        def val_color(self, value: Union[Color, Tuple[int, int, int, int], Iterable[int]]) -> None: ...

        @property
        def val_colour(self) -> Color: return Color(0, 0, 0)
        @val_colour.setter
        def val_colour(self, value: Union[Color, Tuple[int, int, int, int], Iterable[int]]) -> None: ...

        @property
        def val_str(self) -> str: return ''
        @val_str.setter
        def val_str(self, value: object) -> None: ...

        @property
        def val_string(self) -> str: return ''
        @val_string.setter
        def val_string(self, value: object) -> None: ...

        @property
        def val_ang(self) -> FrozenAngle: return FrozenAngle()
        @val_ang.setter
        def val_ang(self, value: Union[Angle, FrozenAngle, Tuple[float, float, float], Iterable[float]]) -> None: ...

        @property
        def val_angle(self) -> FrozenAngle: return FrozenAngle()
        @val_angle.setter
        def val_angle(self, value: Union[Angle, FrozenAngle, Tuple[float, float, float], Iterable[float]]) -> None: ...

        val_quat: Quaternion
        val_quaternion: Quaternion

        @property
        def val_mat(self) -> FrozenMatrix: return FrozenMatrix()
        @val_mat.setter
        def val_mat(self, value: Union[Matrix, FrozenMatrix, Angle, FrozenAngle]) -> None: ...

        @property
        def val_matrix(self) -> FrozenMatrix: return FrozenMatrix()
        @val_matrix.setter
        def val_matrix(self, value: Union[Matrix, FrozenMatrix, Angle, FrozenAngle]) -> None: ...

        val_bytes: bytes
        val_bin: bytes
        val_binary: bytes

        val_compound: _Element
        val_elem: _Element
        val: _Element
    else:
        val_int = _make_val_prop(ValueType.INT, int)
        val_str = val_string = _make_val_prop(ValueType.STRING, str)
        val_bin = val_binary = val_bytes = _make_val_prop(ValueType.BINARY, bytes)
        val_float = _make_val_prop(ValueType.FLOAT, float)
        val_time = _make_val_prop(ValueType.TIME, Time)
        val_bool = _make_val_prop(ValueType.BOOL, bool)
        val_colour = val_color = _make_val_prop(ValueType.COLOR, Color)
        val_vec2 = _make_val_prop(ValueType.VEC2, Vec2)
        val_vec3 = _make_val_prop(ValueType.VEC3, FrozenVec)
        val_vec4 = _make_val_prop(ValueType.VEC4, Vec4)
        val_quat = val_quaternion = _make_val_prop(ValueType.QUATERNION, Quaternion)
        val_ang = val_angle = _make_val_prop(ValueType.ANGLE, FrozenAngle)
        val_mat = val_matrix = _make_val_prop(ValueType.MATRIX, FrozenMatrix)
        val_compound = val_elem = val = _make_val_prop(ValueType.ELEMENT, _Element)

del _make_val_prop


# Uses private parts of Attribute only.
# noinspection PyProtectedMember
class AttrMember(_ValProps):
    """A proxy for individual indexes/keys, allowing having .val attributes."""
    def __init__(self, owner: 'Attribute[Any]', index: int) -> None:
        """Internal use only."""
        self.owner = owner
        self.index = index

    def _read_val(self, newtype: ValueType) -> Value:
        if isinstance(self.owner._value, list):
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
        if isinstance(self.owner._value, list):
            self.owner._value[self.index] = convert
        else:
            self.owner._value = convert


class Attribute(Generic[ValueT], _ValProps):
    """A single attribute of an element.

    Attributes store either a single scalar value, or an array of multiple values of the same type. Creation is accomplished mainly via the many classmethods, one for each value type.

    To access the value, read/write one of the `val_*` properties, which will if possible convert
    the value to the desired type. For arrays either use `iter_*()` or `attr[ind].val_*` to fetch
    by index.
    """
    __slots__ = ['name', '_typ', '_value']
    name: str
    _typ: ValueType
    _value: Union[ValueT, List[ValueT]]

    # Overload with ValueType -> type matchup.
    @overload
    def __init__(self: 'Attribute[Element]', name: str, val_type: Literal[ValueType.ELEMENT], value: Union['Element', List['Element']]) -> None: ...
    @overload
    def __init__(self: 'Attribute[int]', name: str, val_type: Literal[ValueType.INTEGER], value: Union[int, List[int]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[float]', name: str, val_type: Literal[ValueType.FLOAT], value: Union[float, List[float]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[bool]', name: str, val_type: Literal[ValueType.BOOL], value: Union[bool, List[bool]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[str]', name: str, val_type: Literal[ValueType.STR], value: Union[str, List[str]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[bytes]', name: str, val_type: Literal[ValueType.BIN], value: Union[str, List[str]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[Time]', name: str, val_type: Literal[ValueType.TIME], value: Union[Time, List[Time]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[Color]', name: str, val_type: Literal[ValueType.COLOR], value: Union[Color, List[Color]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[Vec2]', name: str, val_type: Literal[ValueType.VEC2], value: Union[Vec2, List[Vec2]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[FrozenVec]', name: str, val_type: Literal[ValueType.VEC3], value: Union[FrozenVec, List[FrozenVec]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[Vec4]', name: str, val_type: Literal[ValueType.VEC4], value: Union[Vec4, List[Vec4]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[FrozenAngle]', name: str, val_type: Literal[ValueType.ANGLE], value: Union[FrozenAngle, List[FrozenAngle]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[Quaternion]', name: str, val_type: Literal[ValueType.QUATERNION], value: Union[Quaternion, List[Quaternion]]) -> None: ...
    @overload
    def __init__(self: 'Attribute[FrozenMatrix]', name: str, val_type: Literal[ValueType.MATRIX], value: Union[FrozenMatrix, List[FrozenMatrix]]) -> None: ...

    @overload
    def __init__(self, name: str, val_type: ValueType, value: Union[ValueT, List[ValueT]]) -> None: ...
    @overload
    def __init__(self, name: str, val_type: ValueType, value: Union[Value, ValueList]) -> None: ...

    def __init__(self, name: str, val_type: ValueType, value: Union[ValueT, List[ValueT]]) -> None:  # type: ignore
        """For internal use only."""
        self.name = name
        self._typ = val_type
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
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.ELEMENT],
        values: Iterable['Element'] = (),
    ) -> 'Attribute[Element]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.INTEGER],
        values: Iterable[builtins.int] = (),
    ) -> 'Attribute[builtins.int]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.FLOAT],
        values: Iterable[builtins.float] = (),
    ) -> 'Attribute[builtins.float]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.BOOL],
        values: Iterable[builtins.bool] = (),
    ) -> 'Attribute[builtins.bool]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.STR],
        values: Iterable[builtins.str] = (),
    ) -> 'Attribute[builtins.str]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.BIN],
        values: Iterable[builtins.bytes] = (),
    ) -> 'Attribute[builtins.bytes]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.TIME],
        values: Iterable[Time] = (),
    ) -> 'Attribute[Time]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.COLOR],
        values: Iterable[Color] = (),
    ) -> 'Attribute[Color]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.VEC2],
        values: Iterable[Vec2] = (),
    ) -> 'Attribute[Vec2]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.VEC3],
        values: Iterable[Union[FrozenVec, Vec, Iterable[float]]] = (),
    ) -> 'Attribute[FrozenVec]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.VEC4],
        values: Iterable[Vec2] = (),
    ) -> 'Attribute[Vec4]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.ANGLE],
        values: Iterable[Union[Angle, FrozenAngle, Matrix, FrozenMatrix, Iterable[float]]] = (),
    ) -> 'Attribute[FrozenAngle]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.QUATERNION],
        values: Iterable[Quaternion] = (),
    ) -> 'Attribute[Quaternion]': ...
    @classmethod
    @overload
    def array(
        cls, name: str,
        val_type: Literal[ValueType.MATRIX],
        values: Iterable[Union[FrozenAngle, Angle, FrozenMatrix, Matrix]] = (),
    ) -> 'Attribute[FrozenMatrix]': ...

    @classmethod
    @overload
    def array(cls, name: str, val_type: ValueType) -> 'Attribute[Any]': ...

    @classmethod
    def array(cls, name: str, val_type: ValueType, values: Iterable[Any]=()) -> 'Attribute[Any]':
        """Create an attribute with an empty array of a specified type."""
        conv_func = CONVERSIONS[val_type]
        return Attribute(name, val_type, list(map(conv_func, values)))

    @classmethod
    def int(cls, name: str, value: Union[builtins.int, List[builtins.int]]) -> 'Attribute[builtins.int]':
        """Create an attribute with an integer value."""
        return Attribute(name, ValueType.INTEGER, value)

    @classmethod
    def float(cls, name: str, value: Union[builtins.float, List[builtins.float]]) -> 'Attribute[builtins.float]':
        """Create an attribute with a float value."""
        return Attribute(name, ValueType.FLOAT, value)

    @classmethod
    def time(cls, name: str, value: Union[Time, builtins.float]) -> 'Attribute[Time]':
        """Create an attribute with a 'time' value.

        This is effectively a float, and only available in binary v3+."""
        return Attribute(name, ValueType.TIME, CONVERSIONS[ValueType.TIME](value))

    @classmethod
    def bool(cls, name: str, value: Union[builtins.bool, List[builtins.bool]]) -> 'Attribute[builtins.bool]':
        """Create an attribute with a boolean value."""
        return Attribute(name, ValueType.BOOL, value)

    @classmethod
    def string(cls, name: str, value: Union[builtins.str, List[builtins.str]]) -> 'Attribute[builtins.str]':
        """Create an attribute with a string value."""
        return Attribute(name, ValueType.STRING, value)

    @classmethod
    def binary(cls, name: str, value: Union[builtins.bytes, List[builtins.bytes]]) -> 'Attribute[builtins.bytes]':
        """Create an attribute with binary data."""
        return Attribute(name, ValueType.BINARY, value)

    @classmethod
    @overload
    def vec2(cls, __name: str, __it: Iterable[builtins.float]) -> 'Attribute[Vec2]': ...
    @classmethod
    @overload
    def vec2(cls, __name: str, x: builtins.float = 0.0, y: builtins.float = 0.0) -> 'Attribute[Vec2]': ...
    @classmethod
    def vec2(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
    ) -> 'Attribute[Vec2]':
        """Create an attribute with a 2D vector."""
        if isinstance(x, (int, float)):
            x_ = float(x)
        else:
            it = iter(x)
            x_ = float(next(it, 0.0))
            y = float(next(it, y))
        return Attribute(name, ValueType.VEC2, Vec2(x_, y))

    @classmethod
    @overload
    def vec3(cls, __name: str, __it: Iterable[builtins.float]) -> 'Attribute[FrozenVec]': ...
    @classmethod
    @overload
    def vec3(
        cls, __name: str,
        x: builtins.float = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> 'Attribute[FrozenVec]': ...
    @classmethod
    def vec3(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]]=0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> 'Attribute[FrozenVec]':
        """Create an attribute with a 3D vector."""
        if isinstance(x, (int, float)):
            x_ = float(x)
        else:
            it = iter(x)
            x_ = float(next(it, 0.0))
            y = float(next(it, y))
            z = float(next(it, z))
        return Attribute(name, ValueType.VEC3, FrozenVec(x_, y, z))

    @classmethod
    @overload
    def vec4(cls, __name: str, __it: Iterable[builtins.float]) -> 'Attribute[Vec4]': ...
    @classmethod
    @overload
    def vec4(
        cls, __name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> 'Attribute[Vec4]': ...
    @classmethod
    def vec4(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> 'Attribute[Vec4]':
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
    @overload
    def color(cls, __name: str, __it: Iterable[Union[builtins.float, builtins.int]]) -> 'Attribute[Color]': ...
    @classmethod
    @overload
    def color(
        cls, __name: str,
        r: Union[builtins.float, builtins.int] = 0,
        g: Union[builtins.float, builtins.int] = 0,
        b: Union[builtins.float, builtins.int] = 0,
        a: Union[builtins.float, builtins.int] = 255,
    ) -> 'Attribute[Color]': ...
    @classmethod
    def color(
        cls, name: str,
        r: Union[builtins.float, builtins.int, Iterable[Union[builtins.float, builtins.int]]]=0,
        g: Union[builtins.float, builtins.int]=0,
        b: Union[builtins.float, builtins.int]=0,
        a: Union[builtins.float, builtins.int]=255,
    ) -> 'Attribute[Color]':
        """Create an attribute with a color."""
        if isinstance(r, (int, float)):
            r_ = r
        else:
            it = iter(r)
            r_ = next(it, 0)
            g = next(it, g)
            b = next(it, b)
            a = next(it, a)
        return Attribute(name, ValueType.COLOR, Color(int(r_), int(g), int(b), int(a)))

    @classmethod
    @overload
    def angle(cls, __name: str, __it: Iterable[builtins.float]) -> 'Attribute[FrozenAngle]': ...
    @classmethod
    @overload
    def angle(
        cls, __name: str,
        pitch: builtins.float = 0.0,
        yaw: builtins.float = 0.0,
        roll: builtins.float = 0.0,
    ) -> 'Attribute[FrozenAngle]': ...
    @classmethod
    def angle(
        cls, name: str,
        pitch: Union[builtins.float, Iterable[builtins.float]]=0.0,
        yaw: builtins.float = 0.0, roll: builtins.float = 0.0,
    ) -> 'Attribute[FrozenAngle]':
        """Create an attribute with an Euler angle."""
        if isinstance(pitch, (int, float)):
            pitch_ = float(pitch)
        else:
            it = iter(pitch)
            pitch_ = float(next(it, 0.0))
            yaw = float(next(it, yaw))
            roll = float(next(it, roll))
        return Attribute(name, ValueType.ANGLE, FrozenAngle(pitch_, yaw, roll))

    @classmethod
    @overload
    def quaternion(cls, __name: str, __it: Iterable[builtins.float]) -> 'Attribute[Quaternion]': ...
    @classmethod
    @overload
    def quaternion(
        cls, __name: str,
        x: builtins.float = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> 'Attribute[Quaternion]': ...
    @classmethod
    def quaternion(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> 'Attribute[Quaternion]':
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

    def _iter_array(self, newtype: ValueType) -> Iterator[Value]:
        """Iterate over the values, converted to the desired type."""
        try:
            func = TYPE_CONVERT[self._typ, newtype]
        except KeyError:
            raise ValueError(
                f'Cannot convert ({self._value!r}) to {newtype} type!')
        if isinstance(self._value, list):
            return map(func, self._value)
        else:
            try:
                return iter([func(self._value)])
            except StructError as exc:
                # Struct errors don't include the value.
                raise ValueError(f'Failed to convert: {exc}, with value {self._value!r}') from None

    if TYPE_CHECKING:
        def iter_int(self) -> Iterator[builtins.int]: ...
        def iter_float(self) -> Iterator[builtins.int]: ...
        def iter_time(self) -> Iterator[Time]: ...
        def iter_bool(self) -> Iterator[builtins.bool]: ...
        def iter_vec2(self) -> Iterator[Vec2]: ...
        def iter_vec3(self) -> Iterator[FrozenVec]: ...
        def iter_vec4(self) -> Iterator[Vec4]: ...
        def iter_color(self) -> Iterator[Color]: ...
        def iter_colour(self) -> Iterator[Color]: ...

        def iter_str(self) -> Iterator[str]: ...
        def iter_string(self) -> Iterator[str]: ...

        def iter_angle(self) -> Iterator[FrozenAngle]: ...
        def iter_ang(self) -> Iterator[FrozenAngle]: ...

        def iter_quat(self) -> Iterator[Quaternion]: ...
        def iter_quaternion(self) -> Iterator[Quaternion]: ...

        def iter_mat(self) -> Iterator[FrozenMatrix]: ...
        def iter_matrix(self) -> Iterator[FrozenMatrix]: ...

        def iter_bytes(self) -> Iterator[bytes]: ...
        def iter_bin(self) -> Iterator[bytes]: ...
        def iter_binary(self) -> Iterator[bytes]: ...

        def iter_compound(self) -> Iterator['Element']: ...
        def iter_elem(self) -> Iterator['Element']: ...
    else:
        iter_int = _make_iter(ValueType.INT, builtins.int)
        iter_str = iter_string = _make_iter(ValueType.STRING, str)
        iter_bin = iter_binary = iter_bytes = _make_iter(ValueType.BINARY, bytes)
        iter_float = _make_iter(ValueType.FLOAT, builtins.float)
        iter_time = _make_iter(ValueType.TIME, Time)
        iter_bool = _make_iter(ValueType.BOOL, builtins.bool)
        iter_colour = iter_color = _make_iter(ValueType.COLOR, Color)
        iter_vec2 = _make_iter(ValueType.VEC2, Vec2)
        iter_vec3 = _make_iter(ValueType.VEC3, FrozenVec)
        iter_vec4 = _make_iter(ValueType.VEC4, Vec4)
        iter_quat = iter_quaternion = _make_iter(ValueType.QUATERNION, Quaternion)
        iter_ang = iter_angle = _make_iter(ValueType.ANGLE, FrozenAngle)
        iter_mat = iter_matrix = _make_iter(ValueType.MATRIX, FrozenMatrix)
        iter_compound = iter_elem = _make_iter(ValueType.ELEMENT, _Element)

    def _write_val(self, newtype: ValueType, value: Value) -> None:
        """Change the type of the atribute."""
        self._typ = newtype
        self._value = CONVERSIONS[newtype](value)

    def __repr__(self) -> str:
        if self._typ is not ValueType.ELEMENT and isinstance(self._value, list) and len(self._value) > 8:
            # Trim down long arrays to make it more readable.
            value = ', '.join(map(repr, self._value[:8]))
            value = f'[{value}, ...]'
        else:
            value = repr(self._value)
        return f'<{self._typ.name} Attr {self.name!r}: {value}>'

    def __eq__(self, other: object) -> builtins.bool:
        if isinstance(other, Attribute):
            return (
                self._typ is other._typ and
                self.name.casefold() == other.name.casefold() and
                self._value == other._value
            )
        return NotImplemented

    def __ne__(self, other: object) -> builtins.bool:
        if isinstance(other, Attribute):
            return (
                self._typ is not other._typ or
                self.name.casefold() != other.name.casefold() or
                self._value != other._value
            )
        return NotImplemented

    def __getitem__(self, item: builtins.int) -> AttrMember:
        """Read values in an array element."""
        if not isinstance(self._value, list):
            raise ValueError('Cannot index singular elements.')
        _ = self._value[item]  # Raise IndexError/KeyError if not present.
        return AttrMember(self, item)

    def __setitem__(self, item: builtins.int, value: ConvValue) -> None:
        """Set a specific array element to a value."""
        if not isinstance(self._value, list):
            raise ValueError('Cannot index singular elements.')
        arr: List[Value] = self._value  # type: ignore
        [val_type, result] = deduce_type_single(value)
        if val_type is not self._typ:
            # Try converting.
            try:
                func = TYPE_CONVERT[val_type, self._typ]
            except KeyError:
                raise ValueError(f'Cannot convert ({val_type}) to {self._typ} type!')
            arr[item] = func(result)
        else:
            arr[item] = result

    def __delitem__(self, item: Union[builtins.int, slice]) -> None:
        """Remove the specified array index(s)."""
        if not isinstance(self._value, list):
            raise ValueError('Cannot index singular elements.')
        del self._value[item]

    def __len__(self) -> builtins.int:
        """Return the number of values in the array, if this is one."""
        if isinstance(self._value, list):
            return len(self._value)
        raise ValueError('Singular elements have no length!')

    def __iter__(self) -> Iterator[ValueT]:
        """Yield each of the elements in an array."""
        warnings.warn("Use explicit attr.iter_X() methods to indicate desired type.", DeprecationWarning, stacklevel=2)
        if isinstance(self._value, list):
            return iter(self._value)
        else:
            return iter((self._value, ))

    def append(self, value: ConvValue) -> None:
        """Append an item to the array.

        If not already an array, it is converted to one
        holding the existing value.
        """
        if not isinstance(self._value, list):
            self._value = cast('List[ValueT]', [self._value])
        [val_type, result] = deduce_type_single(value)
        if val_type is not self._typ:
            # Try converting.
            try:
                func = cast('Callable[[Value], ValueT]', TYPE_CONVERT[val_type, self._typ])
            except KeyError:
                raise ValueError(f'Cannot convert ({val_type}) to {self._typ} type!')
            self._value.append(func(result))
        else:
            self._value.append(result)  # type: ignore # (we know it's right)

    def extend(self, values: Iterable[ConvValue]) -> None:
        """Append multiple values to the array.

        If not already an array, it is converted to one
        holding the existing value.
        """
        if not isinstance(self._value, list):
            self._value = cast('List[ValueT]', [self._value])
        for value in values:
            [val_type, result] = deduce_type_single(value)
            if val_type is not self._typ:
                # Try converting.
                try:
                    func = cast('Callable[[Value], ValueT]', TYPE_CONVERT[val_type, self._typ])
                except KeyError:
                    raise ValueError(f'Cannot convert ({val_type}) to {self._typ} type!')
                self._value.append(func(result))
            else:
                self._value.append(result)  # type: ignore # (we know it's right)

    def clear_array(self) -> None:
        """Remove all items in this, if it is an array."""
        if isinstance(self._value, list):
            self._value.clear()
        else:
            raise ValueError('Singular elements cannot be cleared!')

    def __copy__(self) -> 'Attribute[ValueT]':
        """Duplicate this attribute shallowly, retaining references if this is an Element type."""
        if isinstance(self._value, list):
            return Attribute(self.name, self._typ, self._value.copy())
        else:
            return Attribute(self.name, self._typ, self._value)

    copy = __copy__

    def __deepcopy__(self, memodict: Any=EmptyMapping) -> 'Attribute[ValueT]':
        """Duplicate this attribute and all children."""
        return Attribute(self.name, self._typ, copy.deepcopy(self._value, memodict))


class Element(Mapping[str, Attribute[Any]]):
    """An element in a DMX tree.

    This is a mapping over `Attribute` objects, representing each key-value pair in the element.
    Other than the three special attributes `name`, `type` and `uuid`, keys are accessed by regular
    mapping methods, producing `Attribute` objects. As a convenience, values or lists of values can
    be assigned directly to implicitly create the Attribute.
    """
    __slots__ = ['type', 'uuid', '_members']

    type: str
    """
    In Valve's formats, this is the name of the C++ class that the element should deserialise
    into, like `DMElement` for example. It can be used for any purpose though. In binary files,
    elements with identical types names are deduplicated in the file.
    """
    uuid: UUID
    """
    This used in the serialised file to allow elements to be appear multiple times in the tree,
    including recursively. This does not normally need to be set manually, since a random one 
    will be computed for each new element automatically.
    """
    _members: MutableMapping[str, Attribute[Any]]

    def __init__(self, name: str, type: str, uuid: Optional[UUID] = None) -> None:
        self._members = {'name': Attribute('name', ValueType.STRING, name)}
        self.type = type
        if uuid is None:
            self.uuid = get_uuid()
        else:
            self.uuid = uuid

    @property
    def is_stub(self) -> bool:
        """Check if this is a 'stub' element, found in binary DMXes."""
        return isinstance(self, StubElement) and self._type is _StubType.STUB

    @property
    def is_null(self) -> bool:
        """Check if this is a NULL element, found in binary DMXes."""
        return isinstance(self, StubElement) and self._type is _StubType.NULL

    @property
    def name(self) -> str:
        """The name can be any string, but usually is some sort of item ID.

        For elements which are children of another element, the name usually matches the key.
        This is actually just a key with the name ``name``. If it is missing, ``""`` is returned.
        """
        try:
            return self._members['name'].val_string
        except KeyError:
            return ''

    @name.setter
    def name(self, value: str) -> None:
        try:
            attr = self._members['name']
        except KeyError:
            self._members['name'] = Attribute('name', ValueType.STRING, value)
        else:
            attr._typ = ValueType.STRING
            attr._value = str(value)

    @classmethod
    def parse(cls, file: IO[bytes], unicode: bool = False) -> Tuple['Element', str, int]:
        """Parse a DMX file encoded in binary or KV2 (text).

        The return value is the tree, format name and version. The format name and version
        are stored in the header, and allow validating the kind of dat stored in the file.
        If unicode is set to True, strings will always be treated as UTF8, even if this module's
        encoding name is not provided. Otherwise, only ASCII characters are allowed for safety.
        """
        # The format header is:
        # <!-- dmx encoding [encoding] [version] format [format] [version] -->
        header = bytearray(file.read(256))
        if not header.startswith(b'<!--'):
            raise ValueError('The file is not a DMX file.')

        # To handle bigger headers, read until we find the -->, or until we
        # arbitrarily read a lot of characters (assume the file is corrupt).
        for i in range(32):
            header_len = header.find(b'-->', -260)
            if header_len > 0:
                header_len += 3
                break
            header.extend(file.read(256))
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
                raise ValueError(f'Invalid DMX header {bytes(header[:header_len])!r}!')
            enc_name = match.group(0)
            if enc_name == b'sfm':
                enc_name = b'binary'
            unicode = False
            enc_vers = 0
            fmt_name = ''
            fmt_vers = 0

        # Seek back to where the end of the header is
        file.seek(header_len)

        if enc_name == b'keyvalues2':
            file_txt = io.TextIOWrapper(file, encoding='utf8' if unicode else 'ascii')
            try:
                result = cls.parse_kv2(file_txt, enc_vers)
            finally:
                # The caller opened our file, so we want to return it to their control.
                # If we don't detach or close, we get a ResourceWarning.
                file_txt.detach()
        elif enc_name == b'binary':
            result = cls.parse_bin(file, enc_vers, unicode)
        else:
            raise ValueError(f'Unknown DMX encoding {repr(enc_name)[2:-1]}!')

        return result, fmt_name, fmt_vers

    @classmethod
    def parse_bin(cls, file: IO[bytes], version: int, unicode: bool = False) -> 'Element':
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
            stringdb_size = stringdb_ind = ''

        if stringdb_size:
            [string_count] = binformat.struct_read(stringdb_size, file)
            stringdb = binformat.read_nullstr_array(file, string_count, encoding)
        else:
            stringdb = None

        stubs: Dict[UUID, StubElement] = {}
        attr: Attribute[Any]

        [element_count] = binformat.struct_read('<i', file)
        elements: List[Element] = []
        for _ in range(element_count):
            if stringdb is not None:
                [ind] = binformat.struct_read(stringdb_ind, file)
                el_type = stringdb[ind]
            else:
                el_type = binformat.read_nullstr(file)
            if version >= 4:
                assert stringdb is not None
                [ind] = binformat.struct_read(stringdb_ind, file)
                name = stringdb[ind]
            else:
                name = binformat.read_nullstr(file, encoding=encoding)
            uuid = UUID(bytes_le=file.read(16))
            elements.append(Element(name, el_type, uuid))
        # Now, the attributes in the elements.
        for elem in elements:
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
                    array: List[Any] = []
                    array_iter: Iterable[int]
                    attr = Attribute(name, attr_type, array)
                    if array_size is not None:
                        array_iter = range(array_size)
                    else:
                        array_iter = (0, )  # Single element, run the loop once to reuse code.

                    for _ in array_iter:
                        [ind] = binformat.struct_read('<i', file)
                        if ind == -1:
                            child_elem = NULL
                        elif ind == -2:
                            # Stub element, just with a UUID.
                            uuid = UUID(binformat.read_nullstr(file))
                            try:
                                child_elem = stubs[uuid]
                            except KeyError:
                                child_elem = stubs[uuid] = StubElement.stub(uuid)
                        else:
                            child_elem = elements[ind]
                        array.append(child_elem)
                    if array_size is None:
                        # Unpack into a single element.
                        [attr._value] = array

                elif attr_type is ValueType.STRING:
                    if array_size is not None:
                        # Arrays are always raw ASCII in the file.
                        attr = Attribute.string(name, binformat.read_nullstr_array(file, array_size))
                    else:  # Single string.
                        if version >= 4:
                            assert stringdb is not None
                            [ind] = binformat.struct_read(stringdb_ind, file)
                            value = stringdb[ind]
                        else:
                            # Raw value.
                            value = binformat.read_nullstr(file, encoding=encoding)
                        attr = Attribute.string(name, value)
                elif attr_type is ValueType.BINARY:
                    # Binary blobs.
                    if array_size is not None:
                        array = []
                        attr = Attribute.binary(name, array)
                        for _ in range(array_size):
                            [size] = binformat.struct_read('<i', file)
                            array.append(file.read(size))
                    else:
                        [size] = binformat.struct_read('<i', file)
                        attr = Attribute.binary(name, file.read(size))
                else:
                    # All other types are fixed-length.
                    size = SIZES[attr_type]
                    conv = TYPE_CONVERT[ValueType.BINARY, attr_type]
                    if array_size is not None:
                        file_, size_ = file, size  #  Prevent other uses from being cellvars.
                        attr = Attribute(name, attr_type, [
                            conv(file_.read(size_))
                            for _ in range(array_size)
                        ])
                    else:
                        attr = Attribute(name, attr_type, conv(file.read(size)))
                elem._members[name.casefold()] = attr

        try:
            return elements[0]
        except IndexError:
            raise ValueError("No elements in DMX file!") from None

    @classmethod
    def parse_kv2(cls, file: IO[str], version: int, unicode: bool = False) -> 'Element':
        """Parse a DMX file encoded in KeyValues2.

        The <!-- --> format comment line should have already been read.
        """
        # We apply UUID lookups after everything's parsed.
        id_to_elem: Dict[UUID, Element] = {}

        # Locations in arrays which are UUIDs (and need setting).
        # This is a (attr, index, uuid, line_num) tuple.
        fixups: List[Tuple[Attribute[Any], Optional[int], UUID, int]] = []
        # Ensure these reuse the same objects.
        stubs: Dict[UUID, StubElement] = collections.defaultdict(StubElement.stub)

        elements = []

        tok = Tokenizer(file)
        for token, tok_value in tok:
            if token is Token.STRING:
                elem_name = tok_value
            elif token is Token.NEWLINE:
                continue
            else:
                raise tok.error(token)
            elements.append(cls._parse_kv2_element(tok, id_to_elem, fixups, stubs, '', elem_name))

        for attr, index, uuid, line_num in fixups:
            try:
                elem = id_to_elem[uuid]
            except KeyError:
                continue  # It'll be a stub element.
            if index is None:
                attr._value = elem
            else:
                attr._value[index] = elem

        try:
            return elements[0]
        except IndexError:
            raise tok.error("No elements in DMX file!") from None

    @classmethod
    def _parse_kv2_element(
        cls, tok: Tokenizer,
        id_to_elem: Dict[UUID, 'Element'],
        fixups: List[Tuple[Attribute[Any], Optional[int], UUID, int]],
        stubs: Dict[UUID, 'StubElement'],
        name: str,
        typ_name: str,
    ) -> 'Element':
        """Parse a compound element."""
        attr: Attribute[Any]
        elem: Element = cls(name, typ_name, _UNSET_UUID)

        for attr_name in tok.block(name):
            orig_typ_name = tok.expect(Token.STRING)
            typ_name = orig_typ_name.casefold()

            # The UUID is a special element name/type combo, if it doesn't match it's a regular
            # attr.
            if attr_name == 'id' and typ_name == 'elementid':
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
                        'Element {} attribute must be "{}" type, not "{}"!',
                        'name', 'string', typ_name
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
                    cls._parse_kv2_element(tok, id_to_elem, fixups, stubs, attr_name, orig_typ_name),
                )
                continue
            if is_array:
                array: List[Any] = []
                attr = Attribute(attr_name, attr_type, array)
                tok.expect(Token.BRACK_OPEN)
                for tok_typ, tok_value in tok.skipping_newlines():
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
                                    # If UUID is present, this stub will be overwritten later.
                                    array.append(stubs[uuid])
                                else:
                                    array.append(NULL)
                            else:
                                # Inline compound
                                array.append(cls._parse_kv2_element(tok, id_to_elem, fixups, stubs, attr_name, tok_value))
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
                    else:
                        raise tok.error(tok_typ)
                else:
                    raise tok.error('Unterminated array!')
            elif attr_type is ValueType.ELEMENT:
                # This is a reference to another element.
                uuid_str = tok.expect(Token.STRING)
                attr = Attribute(attr_name, attr_type, NULL)
                if uuid_str:
                    try:
                        uuid = UUID(uuid_str)
                    except ValueError:
                        raise tok.error('Invalid UUID "{}"!', uuid_str)
                    attr.val_elem = stubs[uuid]
                    fixups.append((attr, None, uuid, tok.line_num))
                    # If the element is present, the stub value  will be overwritten after.
                # else: If blank, it's a NULL.
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
        unicode: Literal['ascii', 'format', 'silent']='ascii',
    ) -> None:
        """Write out a DMX tree, using the binary format.

        :param file: A writable binary file to save to.
        :param version: Must be a number from 0-5.
        :param fmt_name: This can be any string, to indicate what contents the file contains.
        :param fmt_ver: This can be a format number, to indicate changes in the format.
        :param unicode: This controls how Unicode characters are handled.

        For the unicode parameter, the following values are permitted:

        - ``"ascii"`` (the default) raises an error if any value is non-ASCII. This
          ensures no encoding issues occur when read by the game.
        - ``"format"`` changes the encoding format to ``"unicode_binary"``, allowing
          the file to be rejected if the game tries to read it and this module's
          parser to automatically switch to Unicode.
        - ``"silent"`` outputs UTF8 without any marker, meaning it could be parsed
          incorrectly by the game or other utilties. This must be parsed with
          ``unicode=True`` to succeed.
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
        stringdb_size: Optional[str]
        stringdb_ind: Optional[str]
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
                if attr.name == 'name':
                    # Has its own special slot.
                    continue
                if stringdb_ind is not None:
                    used_strings.add(attr.name)
                if attr.type is ValueType.TIME and version < 3:
                    raise ValueError('TIME attributes are not permitted before binary v3!')
                elif attr.type is ValueType.ELEMENT:
                    for subelem in attr.iter_elem():
                        if not isinstance(subelem, StubElement) and subelem.uuid not in elem_to_ind:
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
                assert stringdb_ind is not None
                file.write(pack(stringdb_ind, string_to_ind[elem.name]))
            else:
                file.write(elem.name.encode(encoding) + b'\0')
            file.write(elem.uuid.bytes_le)

        # Now, write out all attributes.
        for elem in elements:
            attr_count = len(elem)
            if 'name' in elem._members:
                attr_count -= 1
            file.write(pack('<i', attr_count))
            for attr in elem.values():
                if attr.name == 'name':
                    continue
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
                        assert stringdb_ind is not None
                        file.write(pack(stringdb_ind, string_to_ind[attr.val_str]))
                    else:
                        for text in attr.iter_string():
                            file.write(text.encode(encoding) + b'\0')
                elif attr.type is ValueType.BINARY:
                    for bin_data in attr.iter_binary():
                        file.write(pack('<i', len(bin_data)))
                        file.write(bin_data)
                elif attr.type is ValueType.ELEMENT:
                    for subelem in attr.iter_elem():
                        if subelem is NULL:  # It's a singleton.
                            file.write(pack('<i', -1))
                        elif subelem.is_stub:
                            file.write(pack('<i', -2))
                        else:
                            file.write(pack('<i', elem_to_ind[subelem.uuid]))
                else:
                    # Convert to binary, then write inline.
                    for bin_data in attr.iter_binary():
                        file.write(bin_data)

    def export_kv2(
        self, file: IO[bytes],
        fmt_name: str = 'dmx', fmt_ver: int = 1,
        *,
        flat: bool = False,
        unicode: Literal['ascii', 'format', 'silent'] = 'ascii',
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
            attr: Attribute[Element]
            for attr in elem.values():
                if attr.type is not ValueType.ELEMENT:
                    continue
                # noinspection PyProtectedMember
                for subelem in attr.iter_elem():
                    if isinstance(subelem, StubElement):
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
            if attr.name == 'name':
                continue
            file.write(b'%b"%b" ' % (
                indent_child,
                attr.name.encode(encoding),
            ))
            if attr.is_array:
                file.write(b'"%b_array"\r\n%b[\r\n' % (attr.type.value.encode(encoding), indent_child))
                indent_arr = indent + b'\t\t'
                for i, child in enumerate(attr._value):
                    file.write(indent_arr)
                    if isinstance(child, Element):
                        if child.is_null:
                            file.write(b'"element" ""')
                        elif child.uuid in roots or child.is_stub:
                            file.write(b'"element" "%b"' % str(child.uuid).encode('ascii'))
                        else:
                            child._export_kv2(file, indent_arr, roots, encoding, cull_uuid)
                    else:
                        str_value = cast(str, TYPE_CONVERT[attr.type, ValueType.STRING](child))
                        file.write(b'"%b"' % (str_value.encode(encoding), ))
                    if i == len(attr) - 1:
                        file.write(b'\r\n')
                    else:
                        file.write(b',\r\n')
                file.write(b'%b]\r\n' % (indent_child, ))
            elif isinstance(attr._value, Element):
                child = attr.val_elem
                if child.is_null:
                    file.write(b'"element" ""\r\n')
                elif child.uuid in roots or child.is_stub:
                    file.write(b'"element" "%b"\r\n' % str(child.uuid).encode('ascii'))
                else:
                    child._export_kv2(file, indent_child, roots, encoding, cull_uuid)
                    file.write(b'\r\n')
            else:
                file.write(b'"%b" "%b"\r\n' % (
                    attr.type.value.encode(encoding),
                    attr.val_str.encode(encoding),
                ))
        file.write(indent + b'}')

    @classmethod
    def from_kv1(cls, props: Keyvalues) -> 'Element':
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

        # First, go through to check if we can inline the attributes, or have to nest.
        # If we have duplicates, both types, or any of the reserved names we need to do so.
        leaf_names: Set[str] = set()
        has_leaf = False
        has_block = False
        no_inline = False
        for child in props:
            if child.has_children():
                has_block = True
            else:
                has_leaf = True
                # The names "name" and "subkeys" are reserved, and can't be used as attributes.
                # ID isn't, because it has a unique attr type to distinguish it from a keyvalue.
                if child.name in {'name', 'subkeys'}:
                    no_inline = True
                if child.name in leaf_names:
                    no_inline = True
                else:
                    leaf_names.add(child.name)
        del leaf_names  # Don't need this anymore, discard it.
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

    def to_kv1(self) -> Keyvalues:
        """Convert an element tree containing a KeyValues 1 tree back into a Property.

        These must satisfy the format from_kv1() produces - all elements have the type DmElement,
        all attributes are strings except for the "subkeys" attribute which is an element array.
        """
        if self.type == NAME_KV1_LEAF:
            return Keyvalues(self.name, self['value'].val_str)
        elif self.type == NAME_KV1:
            kv = Keyvalues(self.name, [])
        elif self.type == NAME_KV1_ROOT:
            kv = Keyvalues.root()
        else:
            raise ValueError(f'{self.type}({self.name!r}) is not a KeyValues1 tree!')
        subkeys: Optional[Attribute[Element]] = None
        for attr in self.values():
            if attr.name == 'subkeys':
                if attr.type != ValueType.ELEMENT or not attr.is_array:
                    raise ValueError('"subkeys" must be an Element array!')
                subkeys = attr
            elif attr.name == 'name':
                continue
            else:
                kv.append(Keyvalues(attr.name, attr.val_str))
        if subkeys is not None:
            for elem in subkeys.iter_elem():
                kv.append(elem.to_kv1())
        return kv

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

    def values(self) -> ValuesView[Attribute[Any]]:
        """Return a view of the attributes for this element."""
        return self._members.values()

    def __getitem__(self, name: str) -> Attribute[Any]:
        return self._members[name.casefold()]

    def __delitem__(self, name: str) -> None:
        """Remove the specified attribute."""
        del self._members[name.casefold()]

    def __setitem__(self, name: str, value: Union[Attribute[Any], ConvValue, Sequence[ConvValue]]) -> None:
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

    def pop(self, name: str, default: Union[Attribute[Any], ConvValue, Sequence[ConvValue]] = _UNSET) -> Attribute[Any]:
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

    def popitem(self) -> Tuple[str, Attribute[Any]]:
        """Remove and return a (name, attr) pair as a 2-tuple, or raise KeyError."""
        key, attr = self._members.popitem()
        return (attr.name, attr)

    def setdefault(self, name: str, default: Union[Attribute[Any], ConvValue, Sequence[ConvValue]]) -> Attribute[Any]:
        """Return the specified attribute name.

        If it does not exist, set it using the default and return that.
        """
        key = name.casefold()
        try:
            return self._members[key]
        except KeyError:
            if not isinstance(default, Attribute):
                typ, val = deduce_type(default)
                conv = CONVERSIONS[typ]
                new_def: Union[Value, ValueList]
                if isinstance(val, list):
                    new_def = list(map(conv, val))
                else:
                    new_def = conv(val)
                default = Attribute(name, typ, new_def)
            self._members[key] = default
            return default


class StubElement(Element):
    """In binary DMXes, it is possible to have stub elements representing values excluded from the
    file.

    This always has no members - attempting to add members will fail silently.
    """
    __slots__ = ['_type']
    def __init__(self, typ: _StubType, uuid: Optional[UUID]) -> None:
        """Internal use only."""
        super().__init__('', typ, uuid)
        # This acts always empty, and can be fake-written to.
        self._members = EmptyMapping
        self._type = typ  # Store redundantly so users trying to change this are ignored.

    @classmethod
    def stub(cls, uuid: Optional[UUID] = None) -> 'StubElement':
        """Create a stubbed element reference with the specified UUID."""
        return cls(_StubType.STUB, uuid)

    def __repr__(self) -> str:
        if self._type is _StubType.STUB:
            return f'<Stub Element: {self.uuid.hex}>'
        elif self._type is _StubType.NULL:
            return '<Null Element>'
        else:
            raise AssertionError(self._type)


# Constant for null elements.
NULL = StubElement(_StubType.NULL, UUID(bytes=bytes(16)))
"""
A special constant Element with an all-zero UUID, representing "null" elements in binary DMX 
files. This always has no members.
"""

_NUMBERS = {int, float, bool}
_ANGLES = {Angle, FrozenAngle}
_VEC3S = {Vec, FrozenVec}
_MATRICES = {Matrix, FrozenMatrix}

# Python types to their matching ValueType.
TYPE_TO_VALTYPE: Dict[type, ValueType] = {
    Element: ValueType.ELEMENT,
    StubElement: ValueType.ELEMENT,
    int: ValueType.INT,
    float: ValueType.FLOAT,
    bool: ValueType.BOOL,
    str: ValueType.STRING,
    bytes: ValueType.BINARY,
    # Time: ValueType.TIME,
    Color: ValueType.COLOR,
    Vec2: ValueType.VEC2,
    FrozenVec: ValueType.VEC3,
    Vec4: ValueType.VEC4,
    FrozenAngle: ValueType.ANGLE,
    Quaternion: ValueType.QUATERNION,
    FrozenMatrix: ValueType.MATRIX,
}


def deduce_type(value: Union[ConvValue, Sequence[ConvValue]]) -> Tuple[ValueType, Union[Value, ValueList]]:
    """Convert Python objects to an appropriate ValueType."""
    if isinstance(value, list):
        return deduce_type_array(value)
    try:  # TypeError if cast is invalid.
        return deduce_type_single(cast(ConvValue, value))
    except TypeError:
        if not isinstance(value, Sequence):
            raise
    return deduce_type_array(value)


def deduce_type_array(value: Sequence[ConvValue]) -> Tuple[ValueType, ValueList]:
    """Convert a Python list to an appropriate ValueType."""
    if len(value) == 0:
        raise TypeError('Cannot deduce type for empty list!')
    types = set(map(type, value))
    if len(types) > 1:
        if types <= _NUMBERS:
            # Allow mixing numerics, casting to the largest subset.
            num_values = cast('List[int | bool | float]', value)
            if float in types:
                return ValueType.FLOAT, [float(x) for x in num_values]
            if int in types:
                return ValueType.INTEGER, [int(x) for x in num_values]
            if bool in types:
                return ValueType.BOOL, [bool(x) for x in num_values]
            raise AssertionError('No numbers?', value)
        # Convert mixed mutable/immutable lists.
        elif types == _ANGLES:
            return ValueType.ANGLE, [FrozenAngle(ang) for ang in cast('List[Angle | FrozenAngle]', value)]
        elif types == _VEC3S:
            return ValueType.VEC3, [FrozenVec(vec) for vec in cast('List[Vec | FrozenVec]', value)]
        elif types == _MATRICES:
            return ValueType.MATRIX, [FrozenMatrix(mat) for mat in cast('List[Matrix | FrozenMatrix]', value)]
        # Else, fall through and try iterables.
    else:
        [val_actual_type] = types
        # Transparently freeze mutable classes.
        if val_actual_type is Matrix:
            return ValueType.MATRIX, [mat.freeze() for mat in cast('List[Matrix]', value)]
        if val_actual_type is Angle:
            return ValueType.ANGLE, [ang.freeze() for ang in cast('List[Angle]', value)]
        if val_actual_type is Vec:
            return ValueType.VEC3, [vec.freeze() for vec in cast('List[Vec]', value)]
        elif val_actual_type is Color:
            # Special case, this must be int.
            return ValueType.COLOR, [
                Color(int(r), int(g), int(b), int(a))
                for r, g, b, a in cast('List[Color]', value)
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
                    tuple.__new__(val_actual_type, map(float, val))  # type: ignore
                    for val in value
                ]
            else:  # Direct type, no need to convert.
                return val_type, list(value)  # type: ignore
    # Deduce each type in the array, check they're the same.
    val_iter = iter(value)
    val_type, first = deduce_type_single(next(val_iter))
    result = [first]
    for val in val_iter:
        sec_type, sec_val = deduce_type_single(val)
        if sec_type is not val_type:
            raise TypeError(
                'Arrays must have the same types, or be all numeric. '
                f'Got {val_type.name} and {sec_type.name}.'
            )
        result.append(sec_val)
    return val_type, result  # type: ignore


def deduce_type_single(value: ConvValue) -> Tuple[ValueType, Value]:
    """Convert a scalar value to an appropriate ValueType."""
    if isinstance(value, Matrix):
        return ValueType.MATRIX, value.freeze()
    if isinstance(value, Angle):
        return ValueType.ANGLE, value.freeze()
    if isinstance(value, Vec):
        return ValueType.VEC3, value.freeze()
    try:
        # Match to one of the types directly.
        val_type = TYPE_TO_VALTYPE[type(value)]
    except KeyError:
        # Try iterables.
        pass
    else:
        # NamedTuple, ensure they're a float.
        if isinstance(value, tuple):
            # Type checker doesn't know all value classes take floats.
            return val_type, tuple.__new__(type(value), map(float, value))  # type: ignore
        else:  # No change needed.
            return val_type, value
    raise TypeError(f'Could not deduce value type for {type(value)}.') from None


# All the type converter functions.
# Assign to globals, then _get_converters() will find and store these funcs,
# removing them from globals.
# Conversion to/from strings and binary are required for all types.

_conv_string_to_float = float
_conv_string_to_integer = int
_conv_string_to_time = lambda text: Time(float(text))
_conv_string_to_bool = lambda time: BOOL_LOOKUP[time.casefold()]
_conv_string_to_vec2 = lambda text: Vec2._make(parse_vector(text, 2))
_conv_string_to_vec3 = lambda text: FrozenVec(parse_vector(text, 3))
_conv_string_to_vec4 = lambda text: Vec4._make(parse_vector(text, 4))
_conv_string_to_angle = lambda text: FrozenAngle(parse_vector(text, 3))
_conv_string_to_quaternion = lambda text: Quaternion._make(parse_vector(text, 4))

def _conv_string_to_color(text: str) -> Color:
    """Colors can either have 3 or 4 values."""
    parts = text.split()
    if len(parts) == 3:
        return Color(int(parts[0]), int(parts[1]), int(parts[2]), 255)
    elif len(parts) == 4:
        return Color(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    else:
        raise ValueError(f"'{text}' is not a valid Color!")

_conv_integer_to_string = str
_conv_integer_to_float = float
_conv_integer_to_time = lambda n: Time(float(n))
_conv_integer_to_bool = bool
_conv_integer_to_vec2 = lambda n: Vec2(n, n)
_conv_integer_to_vec3 = lambda n: FrozenVec(n, n, n)
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
_conv_float_to_time = Time
_conv_float_to_vec2 = lambda n: Vec2(n, n)
_conv_float_to_vec3 = lambda n: FrozenVec(n, n, n)
_conv_float_to_vec4 = lambda n: Vec4(n, n, n, n)

_conv_bool_to_integer = int
_conv_bool_to_float = float
_conv_bool_to_string = bool_as_int

_conv_time_to_integer = lambda t: int(t.value)
_conv_time_to_float = lambda t: t.value
_conv_time_to_bool = lambda t: t.value > 0
_conv_time_to_string = lambda t: str(t.value)

_conv_vec2_to_string = lambda v: f'{_fmt_float(v.x)} {_fmt_float(v.y)}'
_conv_vec2_to_bool = lambda v: bool(v.x or v.y)
_conv_vec2_to_vec3 = lambda v: FrozenVec(v.x, v.y, 0.0)
_conv_vec2_to_vec4 = lambda v: Vec4(v.x, v.y, 0.0, 0.0)

_conv_vec3_to_string = lambda v: f'{_fmt_float(v.x)} {_fmt_float(v.y)} {_fmt_float(v.z)}'
_conv_vec3_to_bool = lambda v: bool(v.x or v.y or v.z)
_conv_vec3_to_vec2 = lambda v: Vec2(v.x, v.y)
_conv_vec3_to_vec4 = lambda v: Vec4(v.x, v.y, v.z, 0.0)
_conv_vec3_to_angle = lambda v: FrozenAngle(v.x, v.y, v.z)
_conv_vec3_to_color = lambda v: Color(v.x, v.y, v.z)

_conv_vec4_to_string = lambda v: f'{_fmt_float(v.x)} {_fmt_float(v.y)} {_fmt_float(v.z)} {_fmt_float(v.w)}'
_conv_vec4_to_bool = lambda v: bool(v.x or v.y or v.z or v.w)
_conv_vec4_to_vec3 = lambda v: FrozenVec(v.x, v.y, v.z)
_conv_vec4_to_vec2 = lambda v: Vec2(v.x, v.y)
_conv_vec4_to_quaternion = lambda v: Quaternion(v.x, v.y, v.z, v.w)
_conv_vec4_to_color = lambda v: Color(v.x, v.y, v.z, v.w)

_conv_matrix_to_angle = lambda mat: mat.to_angle().freeze()

_conv_angle_to_string = lambda a: f'{_fmt_float(a.pitch)} {_fmt_float(a.yaw)} {_fmt_float(a.roll)}'
_conv_angle_to_matrix = lambda ang: FrozenMatrix.from_angle(ang)
_conv_angle_to_vec3 = lambda ang: FrozenVec(ang.pitch, ang.yaw, ang.roll)

_conv_color_to_string = lambda col: f'{col.r} {col.g} {col.b} {col.a}'
_conv_color_to_vec3 = lambda col: FrozenVec(col.r, col.g, col.b)
_conv_color_to_vec4 = lambda col: Vec4(col.r, col.g, col.b, col.a)

_conv_quaternion_to_string = lambda quat: f'{_fmt_float(quat.x)} {_fmt_float(quat.y)} {_fmt_float(quat.z)} {_fmt_float(quat.w)}'
_conv_quaternion_to_vec4 = lambda quat: Vec4(quat.x, quat.y, quat.z, quat.w)

# Binary conversions.
_conv_string_to_binary = bytes.fromhex
if sys.version_info >= (3, 8):
    def _conv_binary_to_string(byt: bytes) -> str:
        """Format each byte, seperated by spaces."""
        return byt.hex(' ', 1).upper()
else:
    def _conv_binary_to_string(byt: bytes) -> str:
        """The parameters for bytes.hex aren't available, do it ourselves."""
        return ' '.join([format(x, '02X') for x in byt])


def _binconv_basic(name: str, fmt: str) -> None:
    """Converter functions for a type with a single value."""
    shape = Struct(fmt)

    def unpack(byt: bytes) -> Any:
        """Binary conversion func."""
        [val] = shape.unpack(byt)
        return val
    ns = globals()
    ns['_struct_' + name] = shape
    ns[f'_conv_{name}_to_binary'] = shape.pack
    ns[f'_conv_binary_to_{name}'] = unpack


def _binconv_cls(name: str, fmt: str, Tup: type) -> None:
    """Converter functions for a type matching a namedtuple."""
    shape = Struct(fmt)
    ns = globals()
    ns['_struct_' + name] = shape
    ns[f'_conv_{name}_to_binary'] = lambda val: shape.pack(*val)
    ns[f'_conv_binary_to_{name}'] = lambda byt: Tup(*shape.unpack(byt))

_binconv_basic('integer', '<i')
_binconv_basic('float', '<f')
_binconv_basic('bool', '<?')

_binconv_cls('color', '<4B', Color)
_binconv_cls('angle', '<3f', FrozenAngle)
_binconv_cls('quaternion', '<4f', Quaternion)

_binconv_cls('vec2', '<2f', Vec2)
_binconv_cls('vec3', '<3f', FrozenVec)
_binconv_cls('vec4', '<4f', Vec4)

_struct_time = Struct('<i')
def _conv_time_to_binary(tim: Time) -> bytes:
    """Time is written as a fixed point integer."""
    return _struct_time.pack(int(round(tim.value * 10000.0)))

def _conv_binary_to_time(byt: bytes) -> Time:
    [num] = _struct_time.unpack(byt)
    return Time(num / 10000.0)

_struct_matrix = Struct('<16f')
def _conv_matrix_to_binary(mat: FrozenMatrix) -> bytes:
    """We only set the 3x3 part."""
    return _struct_matrix.pack(
        mat[0, 0], mat[0, 1], mat[0, 2], 0.0,
        mat[1, 0], mat[1, 1], mat[1, 2], 0.0,
        mat[2, 0], mat[2, 1], mat[2, 2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
def _conv_binary_to_matrix(byt: bytes) -> FrozenMatrix:
    """We only set the 3x3 part."""
    data = _struct_matrix.unpack(byt)
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = data[0:3]
    mat[1, 0], mat[1, 1], mat[1, 2] = data[4:7]
    mat[2, 0], mat[2, 1], mat[2, 2] = data[8:11]
    return mat.freeze()


def _conv_string_to_matrix(text: str) -> FrozenMatrix:
    data = parse_vector(text, 16)
    mat = Matrix()
    mat[0, 0], mat[0, 1], mat[0, 2] = data[0:3]
    mat[1, 0], mat[1, 1], mat[1, 2] = data[4:7]
    mat[2, 0], mat[2, 1], mat[2, 2] = data[8:11]
    return mat.freeze()


def _conv_matrix_to_string(mat: FrozenMatrix) -> str:
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
_conv_float = float
_conv_bool = bool


def _conv_time(value: Union[Time, float]) -> Time:
    """We accept integers, floats and actual Time values."""
    if isinstance(value, Time):
        return value
    elif isinstance(value, float):
        return Time(value)
    elif isinstance(value, int):
        return Time(float(value))
    else:
        raise TypeError('Time requires a Time instance, float or integer.')


def _conv_binary(value: Union[bytes, bytearray, memoryview]) -> bytes:
    """We accept any buffer, but not raw integer lengths."""
    return bytes(memoryview(value))


def _converter_ntup(typ: Type[ValueT]) -> Callable[[Union[ValueT, Iterable[float]]], ValueT]:
    """Common logic for named-tuple members."""
    def _convert(value: Union[ValueT, Iterable[float]]) -> ValueT:
        if isinstance(value, typ):  # pyright: ignore
            return value
        else:
            return typ._make(value)  # type: ignore
    return _convert

_conv_vec2 = _converter_ntup(Vec2)
_conv_vec4 = _converter_ntup(Vec4)
_conv_quaternion = _converter_ntup(Quaternion)
del _converter_ntup


def _conv_color(value: Union[Color, Iterable[int]]) -> Color:
    if isinstance(value, Color):
        return value
    try:
        r, g, b, a = value
    except ValueError:
        try:
            r, g, b = value
            a = 255
        except ValueError:
            raise ValueError(f'Color() requires 3 or 4-long iterable, got: {value!r}')
    return Color(r, g, b, a)


def _conv_vec3(value: Union[Vec, FrozenVec, Iterable[float]]) -> FrozenVec:
    """Convert to a 3-vector."""
    if isinstance(value, FrozenVec):
        return value
    else:
        return FrozenVec(*value)


def _conv_angle(value: Union[FrozenAngle, Angle, FrozenMatrix, Matrix, Iterable[float]]) -> FrozenAngle:
    if isinstance(value, FrozenAngle):
        return value
    elif isinstance(value, Angle):
        return value.freeze()
    elif isinstance(value, Matrix) or isinstance(value, FrozenMatrix):
        return value.to_angle().freeze()
    else:
        return FrozenAngle(*value)


def _conv_matrix(value: Union[FrozenAngle, Angle, FrozenMatrix, Matrix]) -> FrozenMatrix:
    if isinstance(value, FrozenMatrix):
        return value
    elif isinstance(value, Matrix):
        return value.freeze()
    elif isinstance(value, Angle) or isinstance(value, FrozenAngle):
        return FrozenMatrix.from_angle(value)
    else:
        raise ValueError('Matrix attributes must be set to a matrix.')


def _conv_element(value: Element) -> Element:
    if not isinstance(value, Element):
        raise ValueError('Expected element, got: ' + repr(value))
    return value

# Gather all these functions, add to the dicts.
TYPE_CONVERT, CONVERSIONS, SIZES = _get_converters()
