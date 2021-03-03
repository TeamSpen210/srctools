"""There's a lot of complexity with the value properties,
so it's better done in type stub form.
"""
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Type, Generic, Iterable, NewType,
    Dict, Tuple, Callable,
)

from srctools import binformat, bool_as_int, Vec, BOOL_LOOKUP, Matrix, Angle
import builtins

from srctools import Vec_tuple as Vec3  # Re-export.

class ValueType(Enum):
    UNKNOWN = 'unknown'
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
    x: float
    y: float


class Vec4(NamedTuple):
    x: float
    y: float
    z: float
    w: float


class Quaternion(NamedTuple):
    x: float
    y: float
    z: float
    w: float


class Color(NamedTuple):
    r: float
    g: float
    b: float


class AngleTup(NamedTuple):
    pitch: float
    yaw: float
    roll: float

Time = NewType('Time', float)
Value = Union[
    int, float, bool, str, bytes,
    Color,
    NewType,
    Vec2, Vec3,
    Vec4,
    AngleTup,
    Quaternion,
    Matrix,
    Element,
]

ValueT = TypeVar(
    'ValueT',
    int, float, bool, str, bytes,
    Color,
    Vec2, Vec3, Vec4,
    Angle,
    Quaternion,
    Matrix,
    Element,
)

_CONVERSIONS: Dict[Tuple[ValueType, ValueType], Callable[[Value], Value]]


class Element(Generic[ValueT]):
    """An element in a DMX tree."""
    typ: ValueType
    _value: Value

    # These are all properties, but no need to annotate like that.
    val_int: int
    val_float: float
    val_bool: bool

    val_vec2: Vec2
    val_vec3: Vec3
    val_vec4: Vec4

    val_color: Color
    val_colour: Color

    val_str: str
    val_string: str

    val_angle: AngleTup
    val_ang: AngleTup

    val_quat: Quaternion
    val_quaternion: Quaternion

    val_mat: Matrix
    val_matrix: Matrix

    def __init__(self, name: str, typ: ValueType=ValueType.VOID, val: Value=None) -> None: ...

    @classmethod
    def int(cls, name: str, value: builtins.int) -> Element[builtins.int]: ...
    @classmethod
    def float(cls, name: str, value: builtins.float) -> Element[builtins.float]: ...
    @classmethod
    def bool(cls, name: str, value: builtins.bool) -> Element[builtins.bool]: ...
    @classmethod
    def string(cls, name: str, value: builtins.str) -> Element[builtins.str]: ...
    @classmethod
    def binary(cls, name: str, value: builtins.bytes) -> Element[builtins.bytes]: ...

    @classmethod
    def vec2(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
    ) -> Element[Vec2]: ...

    @classmethod
    def vec3(
        cls,
        name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> Element[Vec3]: ...

    @classmethod
    def vec4(
        cls,
        name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> Element[Vec4]: ...

    @classmethod
    def color(
        cls,
        name: str,
        r: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        g: builtins.float = 0.0,
        b: builtins.float = 0.0,
    ) -> 'Element[Color]': ...

    @classmethod
    def angle(
        cls,
        name: str,
        pitch: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        yaw: builtins.float = 0.0,
        roll: builtins.float = 0.0,
    ) -> Element[Vec4]: ...

    @classmethod
    def quaternion(
        cls,
        name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> Element[Quaternion]: ...

    def _read_val(self, newtype: ValueType) -> Value: ...
    def _write_val(self, newtype: ValueType, value: Value) -> None: ...

    def set_val_void(self) -> None: ...
