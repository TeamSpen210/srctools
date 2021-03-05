"""There's a lot of complexity with the value properties,
so it's better done in type stub form.
"""
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Type, Generic, Iterable, NewType,
    Dict, Tuple, Callable, List, IO, overload,
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


VAL_TYPE_TO_IND: Dict[ValueType, int]
IND_TO_VALTYPE: Dict[int, ValueType]
ARRAY_OFFSET: int


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
    r: int
    g: int
    b: int
    a: int


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

TYPE_CONVERT: Dict[Tuple[ValueType, ValueType], Callable[[Value], Value]]
CONVERSIONS: Dict[ValueType, Callable[[object], Value]]
SIZES: Dict[ValueType, int]

def parse_vector(text: str, count: int) -> List[float]: ...


class _ValProps:
    """Properties which read/write as the various kinds of value types."""
    @property
    def val_int(self) -> int: ...
    @val_int.setter
    def val_int(self, value: Union[int, float]): ...

    @property
    def val_float(self) -> int: ...
    @val_float.setter
    def val_float(self, value: float): ...

    @property
    def val_bool(self) -> bool: ...
    @val_bool.setter
    def val_bool(self, value: object): ...

    @property
    def val_vec2(self) -> Vec2: ...
    @val_vec2.setter
    def val_vec2(self, value: Union[Vec2, Tuple[float, float], Iterable[float]]) -> None: ...

    @property
    def val_vec3(self) -> Vec3: ...
    @val_vec3.setter
    def val_vec3(self, value: Union[Vec3, Tuple[float, float, float], Iterable[float]]) -> None: ...

    @property
    def val_vec4(self) -> Vec4: ...
    @val_vec4.setter
    def val_vec4(self, value: Union[Vec4, Tuple[float, float, float, float], Iterable[float]]) -> None: ...

    @property
    def val_color(self) -> Color: ...
    @val_color.setter
    def val_color(self, value: Union[Color, Tuple[int, int, int, int], Iterable[int]]) -> None: ...

    @property
    def val_colour(self) -> Color: ...
    @val_colour.setter
    def val_colour(self, value: Union[Color, Tuple[int, int, int, int], Iterable[int]]) -> None: ...

    @property
    def val_str(self) -> str: ...
    @val_str.setter
    def val_colour(self, value: object) -> None: ...

    @property
    def val_string(self) -> str: ...
    @val_string.setter
    def val_string(self, value: object) -> None: ...

    val_angle: AngleTup
    val_ang: AngleTup

    val_quat: Quaternion
    val_quaternion: Quaternion

    val_mat: Matrix
    val_matrix: Matrix

class ElemMember(_ValProps):
    def __init__(self, owner: Element, index: Union[int, str]) -> None: ...


class Element(Generic[ValueT], _ValProps):
    type: str
    name: str
    uuid: UUID
    _val_typ: Union[ValueType, str]
    _value: Union[Value, list, dict]

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
        r: Union[builtins.int, Iterable[builtins.int]] = 0,
        g: builtins.int = 0,
        b: builtins.int = 0,
        a: builtins.int = 0,
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

    def __repr__(self) -> str: ...

    @overload
    def __getitem__(self: Element[Element], item: Union[str, int]) -> Element: ...
    @overload
    def __getitem__(self, item: Union[str, int]) -> ElemMember: ...

    @overload
    def __setitem__(self: Element[Element], item: Union[str, int], value: Value) -> None: ...
    @overload
    def __setitem__(self: Element[ValueT], item: Union[str, int], value: ValueT) -> None: ...
