"""There's a lot of complexity with the value properties,
so it's better done in type stub form.
"""
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Type, Generic, Iterable, NewType,
    Dict, Tuple, Callable, List, IO
)
from uuid import UUID
from srctools import binformat, bool_as_int, Vec, BOOL_LOOKUP, Matrix, Angle
import builtins

from srctools import Vec_tuple as Vec3  # Re-export.
from srctools.tokenizer import Tokenizer


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
SIZES: Dict[ValueType, int]

def parse_vector(text: str, count: int) -> List[float]: ...

class Element(Generic[ValueT]):
    """An element in a DMX tree."""
    name: str
    typ: ValueType
    _val_typ: Union[Value, list, dict]
    uuid: UUID

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

    def __init__(self, el_type: str, typ: ValueType, val, uuid: UUID=None, name: str='') -> None: ...

    @classmethod
    def parse(cls, file: IO[bytes]) -> Tuple[Element, str, int]: ...
    @classmethod
    def parse_bin(cls, file: IO[bytes], version: int) -> Element: ...
    @classmethod
    def parse_kv2(cls, file: IO[str], version: int) -> Element: ...

    @classmethod
    def _parse_kv2_element(cls, tok: Tokenizer, id_to_elem: Dict[UUID, Element], name: str) -> Element: ...

    @classmethod
    def int(cls, el_type: str, value: builtins.int, name: str='') -> Element[builtins.int]: ...
    @classmethod
    def float(cls, el_type: str, value: builtins.float, name: str='') -> Element[builtins.float]: ...
    @classmethod
    def bool(cls, el_type: str, value: builtins.bool, name: str='') -> Element[builtins.bool]: ...
    @classmethod
    def string(cls, el_type: str, value: builtins.str, name: str='') -> Element[builtins.str]: ...
    @classmethod
    def binary(cls, el_type: str, value: builtins.bytes, name: str='') -> Element[builtins.bytes]: ...

    @classmethod
    def vec2(
        cls, el_type: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        name: str='',
    ) -> Element[Vec2]: ...

    @classmethod
    def vec3(
        cls,
        el_type: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        name: str='',
    ) -> Element[Vec3]: ...

    @classmethod
    def vec4(
        cls,
        el_type: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
        name: str='',
    ) -> Element[Vec4]: ...

    @classmethod
    def color(
        cls,
        el_type: str,
        r: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        g: builtins.float = 0.0,
        b: builtins.float = 0.0,
        name: str='',
    ) -> 'Element[Color]': ...

    @classmethod
    def angle(
        cls,
        el_type: str,
        pitch: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        yaw: builtins.float = 0.0,
        roll: builtins.float = 0.0,
        name: str='',
    ) -> Element[Vec4]: ...

    @classmethod
    def quaternion(
        cls,
        el_type: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
        name: str='',
    ) -> Element[Quaternion]: ...

    def _read_val(self, newtype: ValueType) -> Value: ...
    def _write_val(self, newtype: ValueType, value: Value) -> None: ...

    def set_val_void(self) -> None: ...
