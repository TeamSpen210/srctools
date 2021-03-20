"""There's a lot of complexity with the value properties,
so it's better done in type stub form.
"""
from enum import Enum
from typing import (
    Union, NamedTuple, TypeVar, Generic, Iterable, NewType, Literal,
    Dict, Tuple, Callable, List, IO, Mapping, Optional, overload, Iterator,
)
from uuid import UUID
from srctools import Matrix, Angle
import builtins

from srctools import Vec_tuple as Vec3  # Re-export.
from srctools.tokenizer import Tokenizer


class ValueType(Enum):
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
    Color, Time,
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
    Color, Time,
    Vec2, Vec3, Vec4,
    Angle,
    Quaternion,
    Matrix,
    Element,
)

TYPE_CONVERT: Dict[Tuple[ValueType, ValueType], Callable[[Value], Value]]
CONVERSIONS: Dict[ValueType, Callable[[object], Value]]
SIZES: Dict[ValueType, int]
TYPE_TO_VALTYPE: Dict[type, ValueType]

def parse_vector(text: str, count: int) -> List[float]: ...

@overload
def deduce_type(value: List[bool]) -> Tuple[ValueType, List[bool]]: ...
@overload
def deduce_type(value: List[Union[int, bool]]) -> Tuple[ValueType, List[int]]: ...
@overload
def deduce_type(value: List[Union[int, float, bool]]) -> Tuple[ValueType, List[float]]: ...
@overload
def deduce_type(value: List[object]) -> Tuple[ValueType, List[ValueT]]: ...
@overload
def deduce_type(value: List[Iterable[float]]) -> Tuple[ValueType, List[tuple]]: ...
@overload
def deduce_type(value: List[tuple]) -> Tuple[ValueType, List[tuple]]: ...
@overload
def deduce_type(value: List[Union[Angle, AngleTup]]) -> Tuple[ValueType, AngleTup]: ...
@overload
def deduce_type(value: List[ValueT]) -> Tuple[ValueType, ValueT]: ...

@overload
def deduce_type(value: Union[Angle, AngleTup]) -> Tuple[ValueType, AngleTup]: ...
@overload
def deduce_type(value: ValueT) -> Tuple[ValueType, ValueT]: ...

@overload
def deduce_type_array(value: List[bool]) -> Tuple[ValueType, List[bool]]: ...
@overload
def deduce_type_array(value: List[Union[int, bool]]) -> Tuple[ValueType, List[int]]: ...
@overload
def deduce_type_array(value: List[Union[int, float, bool]]) -> Tuple[ValueType, List[float]]: ...
@overload
def deduce_type_array(value: List[object]) -> Tuple[ValueType, List[ValueT]]: ...
@overload
def deduce_type_array(value: List[Iterable[float]]) -> Tuple[ValueType, List[tuple]]: ...
@overload
def deduce_type_array(value: List[tuple]) -> Tuple[ValueType, List[tuple]]: ...
@overload
def deduce_type_array(value: List[Union[Angle, AngleTup]]) -> Tuple[ValueType, AngleTup]: ...
@overload
def deduce_type_array(value: List[ValueT]) -> Tuple[ValueType, ValueT]: ...

@overload
def deduce_type_single(value: Union[Angle, AngleTup]) -> Tuple[ValueType, AngleTup]: ...
@overload
def deduce_type_single(value: ValueT) -> Tuple[ValueType, ValueT]: ...


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

    val_bytes: bytes
    val_bin: bytes
    val_binary: bytes

    val_compound: Element
    val_elem: Element
    val: Element

class AttrMember(_ValProps):
    def __init__(self, owner: Attribute, index: Union[int, str]) -> None: ...


class Attribute(Generic[ValueT], _ValProps):
    name: str
    _typ: ValueType
    _value: Union[Value, list]

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

    val_elem: Element
    val_compound: Element

    # Readonly
    @property
    def type(self) -> ValueType: ...

    @overload
    def __init__(self: Attribute[Element], name: str, val_type: Literal[ValueType.ELEMENT], value: Union[Element, List[Element]]) -> None: ...
    @overload
    def __init__(self: Attribute[int], name: str, val_type: Literal[ValueType.INTEGER], value: Union[int, List[int]]) -> None: ...
    @overload
    def __init__(self: Attribute[float], name: str, val_type: Literal[ValueType.FLOAT], value: Union[float, List[float]]) -> None: ...
    @overload
    def __init__(self: Attribute[bool], name: str, val_type: Literal[ValueType.BOOL], value: Union[bool, List[bool]]) -> None: ...
    @overload
    def __init__(self: Attribute[str], name: str, val_type: Literal[ValueType.STR], value: Union[str, List[str]]) -> None: ...
    @overload
    def __init__(self: Attribute[bytes], name: str, val_type: Literal[ValueType.BIN], value: Union[str, List[str]]) -> None: ...
    @overload
    def __init__(self: Attribute[Time], name: str, val_type: Literal[ValueType.TIME], value: Union[Time, List[Time]]) -> None: ...
    @overload
    def __init__(self: Attribute[Color], name: str, val_type: Literal[ValueType.COLOR], value: Union[Color, List[Color]]) -> None: ...
    @overload
    def __init__(self: Attribute[Vec2], name: str, val_type: Literal[ValueType.VEC2], value: Union[Vec2, List[Vec2]]) -> None: ...
    @overload
    def __init__(self: Attribute[Vec3], name: str, val_type: Literal[ValueType.VEC3], value: Union[Vec3, List[Vec3]]) -> None: ...
    @overload
    def __init__(self: Attribute[Vec4], name: str, val_type: Literal[ValueType.VEC4], value: Union[Vec4, List[Vec4]]) -> None: ...
    @overload
    def __init__(self: Attribute[AngleTup], name: str, val_type: Literal[ValueType.ANGLE], value: Union[AngleTup, List[AngleTup]]) -> None: ...
    @overload
    def __init__(self: Attribute[Quaternion], name: str, val_type: Literal[ValueType.QUATERNION], value: Union[Quaternion, List[Quaternion]]) -> None: ...
    @overload
    def __init__(self: Attribute[Matrix], name: str, val_type: Literal[ValueType.MATRIX], value: Union[Matrix, List[Matrix]]) -> None: ...

    @overload
    def __init__(self, name: str, type: ValueType, value: Union[ValueT, List[ValueT]]) -> None: ...

    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.ELEMENT]) -> Attribute[Element]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.INTEGER]) -> Attribute[int]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.FLOAT]) -> Attribute[float]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.BOOL]) -> Attribute[bool]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.STR]) -> Attribute[str]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.BIN]) -> Attribute[bytes]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.TIME]) -> Attribute[Time]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.COLOR]) -> Attribute[Color]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.VEC2]) -> Attribute[Vec2]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.VEC3]) -> Attribute[Vec3]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.VEC4]) -> Attribute[Vec4]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.ANGLE]) -> Attribute[AngleTup]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.QUATERNION]) -> Attribute[Quaternion]: ...
    @classmethod
    @overload
    def array(cls, name: str, val_type: Literal[ValueType.MATRIX]) -> Attribute[Matrix]: ...

    @classmethod
    @overload
    def array(cls, name: str, val_type: ValueType) -> Attribute: ...

    @classmethod
    def int(cls, name: str, value: builtins.int) -> Attribute[builtins.int]: ...
    @classmethod
    def float(cls, name: str, value: builtins.float) -> Attribute[builtins.float]: ...
    @classmethod
    def bool(cls, name: str, value: builtins.bool) -> Attribute[builtins.bool]: ...
    @classmethod
    def string(cls, name: str, value: builtins.str) -> Attribute[builtins.str]: ...
    @classmethod
    def binary(cls, name: str, value: builtins.bytes) -> Attribute[builtins.bytes]: ...

    @classmethod
    def vec2(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
    ) -> Attribute[Vec2]: ...

    @classmethod
    def vec3(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
    ) -> Attribute[Vec3]: ...

    @classmethod
    def vec4(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> Attribute[Vec4]: ...

    @classmethod
    def color(
        cls, name: str,
        r: Union[builtins.int, Iterable[builtins.int]] = 0,
        g: builtins.int = 0,
        b: builtins.int = 0,
        a: builtins.int = 0,
    ) -> 'Attribute[Color]': ...

    @classmethod
    def angle(
        cls, name: str,
        pitch: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        yaw: builtins.float = 0.0,
        roll: builtins.float = 0.0,
    ) -> Attribute[Vec4]: ...

    @classmethod
    def quaternion(
        cls, name: str,
        x: Union[builtins.float, Iterable[builtins.float]] = 0.0,
        y: builtins.float = 0.0,
        z: builtins.float = 0.0,
        w: builtins.float = 0.0,
    ) -> Attribute[Quaternion]: ...

    def _read_val(self, newtype: ValueType) -> Value: ...
    def _write_val(self, newtype: ValueType, value: Value) -> None: ...

    def __repr__(self) -> str: ...

    def __getitem__(self, item: int) -> AttrMember: ...
    def __setitem__(self, item: Union[int, slice], value: ValueT) -> None: ...
    def __delitem__(self, item: Union[int, slice]) -> None: ...
    def __len__(self) -> builtins.int: ...
    def __iter__(self) -> Iterator[ValueT]: ...


class Element(Mapping[str, Attribute]):
    name: str
    type: str
    uuid: UUID
    _members: Dict[str, Attribute]

    def __init__(self, name: str, type: str, uuid: UUID=None) -> None: ...

    @classmethod
    def parse(cls, file: IO[bytes]) -> Tuple[Element, str, int]: ...
    @classmethod
    def parse_bin(cls, file: IO[bytes], version: int) -> Element: ...
    @classmethod
    def parse_kv2(cls, file: IO[str], version: int) -> Element: ...

    @classmethod
    def _parse_kv2_element(
        cls, tok: Tokenizer,
        id_to_elem: Dict[UUID, Element],
        fixups: List[Tuple[Attribute, Optional[int], UUID, int]],
        name: str,
        typ_name: str,
    ) -> Element: ...

    def __repr__(self) -> str: ...

    def __getitem__(self, item: str) -> Attribute: ...
    def __setitem__(self, item: str, value: ValueT) -> None: ...
